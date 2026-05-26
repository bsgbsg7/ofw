# Q-Modulation 训练数据管线说明

> 以 `train_stage1.py` 为例，说明训练时数据如何产生、流经哪些模块、最终如何计算损失。

---

## 1. 核心结论：所有数据均为在线合成

Q-Modulation 模型的训练**不使用任何外部数据集**。每一轮训练的全部数据（比特、信道、噪声）都是在 `call()` 内部通过随机过程实时生成的：

| 数据 | 来源 | 说明 |
|------|------|------|
| 发送比特 `b` | `BinarySource()` | 均匀分布的随机 0/1 比特 |
| 多径信道 `a, tau` | `tdl_randomDS` (TDL-A) | 3GPP TDL 信道模型，延迟扩展在 [10ns, 600ns] 内随机采样 |
| AWGN 噪声 | `ApplyTimeChannel` 内部 | 根据当前 Eb/N0 自动添加高斯白噪声 |

这意味着每个 batch 看到的比特和信道都是全新的，不存在 "过拟合固定数据集" 的问题。

---

## 2. 训练循环概览 (`train_stage1.py`)

```
for i in range(5000):
    1. 随机采样 Eb/N0 ∈ [EBN0_DB_MIN+10, EBN0_DB_MAX] = [10, 25] dB
    2. 调用 model_train(batch_size=10*256, ebno_db)
    3. 前向传播 → 计算 total_loss
    4. 反向传播 → 更新网络参数
    5. 每 100 步评估 BER@20dB
```

关键超参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `BATCH_SIZE * 256` | 2560 | 每步训练的样本数 |
| `LEARNING_RATE` | 0.0005 | Adam 优化器学习率 |
| `EBN0_DB_MIN + 10` | 10 dB | SNR 采样下界 |
| `EBN0_DB_MAX` | 25 dB | SNR 采样上界 |

---

## 3. 前向传播完整数据流

下面是 `qQ_MODEL.call()` 中一次前向传播的完整管线，从上到下依次经过 10 个阶段。

### 阶段 1: 随机比特生成

```python
b = self._binary_source([batch_size, 1, 1, 208])
# 形状: [2560, 1, 1, 208]
# 内容: 均匀随机 0/1，共 208 bit = 52 个数据子载波 × 4 bit/symbol (16QAM)
```

`_n = num_data_symbols × NUM_BITS_PER_SYMBOL`

在本配置中：`FFT_SIZE=32, pilot 占 1 个 OFDM symbol`，因此：
- 数据 OFDM 符号数 = `NUM_OFDM_SYMBOL - len(pilot_indices)` = `3 - 1` = 2
- 数据子载波数 = 2 × 32 = 64
- 实际有效数据符号 = `TOT_SYMBOLS_TO_DELIVER = 64` → 实际传输 64 个 QAM 符号
- 比特数 = 64 × 4 = 256（源码中 `_n` 由 `rg.num_data_symbols * 4` 决定，约为 256）

### 阶段 2: QAM 映射

```python
x = self._mapper(b)       # [2560, 1, 1, 256] → [2560, 1, 1, 64] 复数 QAM 符号
x_rg = self._rg_mapper(x) # [2560, 1, 1, num_ofdm_symbols=3, fft_size=32]
```

16QAM: 每 4 个 bit 映射为一个复数星座点。

### 阶段 3: 信道生成（获得真实 CSI）

```python
a, tau = self._channel_model(batch_size, num_time_samples + l_tot - 1, bandwidth)
```

`channel_model = tdl_randomDS`，即 **TDL-A 模型，延迟扩展在 [10ns, 600ns] 随机采样**。

输出：
- `a` — 各路径的复增益，形状 `[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
- `tau` — 各路径的时延

然后转换为两种表示：

```python
# 时域信道：用于实际卷积
h_time = cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=True)

# 频域信道：用于可视化对比
h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)
```

### 阶段 4: CSI 获取（Delta 脉冲法）

这一步是获取 CSI 的关键——直接发送一个 delta 脉冲通过信道，得到信道冲激响应：

```python
# 构造频域 delta 脉冲：[1, 0, 0, ..., 0]
delta = one_hot(0, fft_size)  # 第一个子载波为 1+0j

# 通过时域信道
y_time = channel_time(delta_padded, h_time, noise=0)

# 提取 CIR 抽头 → 这就是神经网络的 CSI 输入
pilots_post_channel = y_time[:, 0, 0, :l_max]  # 形状: [2560, 21, 1]
```

**为什么用 delta 脉冲而不是完整的 OFDM 接收链路？**
- delta 脉冲通过信道后直接等于信道冲激响应，无需 LS 估计、无需 LMMSE 均衡
- 避免了信道估计误差对 Q 矩阵生成的影响
- 计算量更小

### 阶段 5: 计算 RMS 延迟扩展

```python
power = |pilots_post_channel|^2
mean_delay = sum(delay * power) / sum(power)
rms_ds = sqrt( sum(power * (delay - mean_delay)^2) / sum(power) )
```

RMS 延迟扩展是衡量信道频率选择性的关键指标：
- 值小 → 频率平坦衰落 → 通信容易
- 值大 → 频率选择性衰落 → 通信困难，但 PAPR 可能相对不重要

### 阶段 6: Q 矩阵生成（神经网络）

```python
Q, q = qQ_creator_conv_gru(pilots_post_channel, training=True)
# Q: [2560, 32, 32] — N×N 复数矩阵，替代 IFFT
# q: [2560, 32]    — N 维复数向量，用于频域均衡
```

`qQ_creator_conv_gru` 的架构：

```
CIR [B, 21, 1] 
  → Complex Conv1D (64 filters, kernel=32) + BN + ReLU
  → Complex GRU (1024 units) 
  → Feedforward (1024 → 1024) + residual
  → Global Average Pooling
  → Dense(32*32 + 32) × 2  (分别输出 Q 和 q)
  → 归一化: Q / sqrt(trace(Q*Q^H) / N)
```

Q 矩阵的输出经过功率归一化：`Q_normalized = Q / sqrt(trace(Q*Q^H)) * sqrt(N)`，这保证了 Q 调制不改变信号的总功率。

### 阶段 7: Q 调制（替代 IFFT）

```python
x_time = Q_modulator(Q, x_rg)
# 数学等价于: x_time = einsum('bxyzi, bij -> bxyzj', x_rg, Q)
# 对每个 OFDM 符号的每个子载波，用 Q 矩阵的行向量做加权和
```

传统 OFDM 中 IFFT 的公式是：
```
x[n] = (1/sqrt(N)) * sum_{k} X[k] * exp(j*2π*k*n/N)
```

Q 调制将其替换为：
```
x[n] = sum_{k} X[k] * Q[n, k]
```

其中 `Q[n, k]` 是网络学习的复数权重，本质上是 **可学习的基函数**。

### 阶段 8: 时域信道传输

```python
y_time = channel_time(x_time, h_time, no)
# y[t] = sum_{l} h[l] * x[t-l] + noise
```

`no` 由 `ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)` 计算，即将 Eb/N0 转换为噪声方差。

### 阶段 9: Q 解调（替代 FFT）+ 均衡 + 解映射

```python
# Q 解调: r_freq = einsum('bxyzi, bij -> bxyzj', y_time, Q^H)
r_freq = Q_demodulator(Q, y_time)

# 频域均衡: 逐元素乘以 q 向量（类似于单抽头均衡器）
r_freq_equalized = r_freq * q

# 软解调: 均衡后的符号 → LLR (Log-Likelihood Ratio)
llr = demapper(r_freq_equalized, no)
```

### 阶段 10: 损失计算

三个损失项，由 Uncertainty 网络自动加权：

```python
# BCE 损失：通信误码率
bce_loss = BinaryCrossentropy(b, llr)

# PAPR 损失：时域信号峰均比
PAR = emprical_papr(x_time, par_lim)

# 总损失 = w_bce * BCE + w_par * PAPR
total_loss = exp(-log_sigma_bce) * bce_loss 
           + exp(-log_sigma_par) * PAR
           + log_sigma_bce + log_sigma_par  # 正则项
```

**Uncertainty 网络的作用**：根据 RMS 延迟扩展，对每个样本自适应调整 BCE 和 PAPR 的权重。原理是：
- 平坦信道（小 DS）→ PAPR 更重要，因为此时通信本身已经很简单
- 频率选择性信道（大 DS）→ BCE 更重要，因为保证通信质量是首要任务

---

## 4. 训练 & 评估的数据差异

| | 训练 `model_train(batch_size, ebno)` | 评估 `model_eval(200, 20.0)` |
|---|---|---|
| SNR | 在 [10, 25] dB 随机采样 | 固定 20 dB |
| 比特 | 每步随机 | 每步随机 |
| 信道 | 每步随机 | 每步随机 |
| Batch size | 2560 | 200 |
| 输出 | `(total_loss, PAR, bce)` | `(b, b_hat)` |
| 梯度 | 有（`GradientTape`） | 无（`training=False`） |

---

## 5. 为什么不需要离线数据集？

这是一个 **端到端物理层学习** 问题，目标不是拟合某个固定的输入输出映射，而是：

1. **学习最优的时频变换**：Q 矩阵应该使信号在多径信道中传输后，能通过 q 向量简单均衡
2. **在通信质量和 PAPR 之间找到最优平衡**：由 Uncertainty 网络根据信道条件动态调整

因此，只要信道模型（TDL-A with random delay spread）能够覆盖实际部署场景的信道分布，在线生成的数据就足够训练出一个泛化能力强的模型。这类似于 GAN 的训练方式——数据是无穷无尽的。

---

## 6. 数据维度速查表

| 符号 | 形状 | 含义 |
|------|------|------|
| `b` | `[B, 1, 1, K]` | 随机比特，K ≈ 256 |
| `x` | `[B, 1, 1, S]` | QAM 符号，S = K/4 = 64 |
| `x_rg` | `[B, 1, 1, M, N]` | 资源网格，M=3 个 OFDM 符号，N=32 子载波 |
| `a, tau` | `[B, rx, rx_ant, tx, tx_ant, paths, T]` | 多径增益和时延 |
| `h_time` | `[B, rx, rx_ant, tx, tx_ant, taps]` | 时域信道抽头 |
| `pilots_post_channel` | `[B, taps, 1]` | 信道冲激响应（CSI 输入） |
| `rms_ds` | `[B, 1]` | RMS 延迟扩展 |
| `Q` | `[B, N, N]` | 学习的调制矩阵 |
| `q` | `[B, N]` | 学习的均衡向量 |
| `x_time` | `[B, 1, 1, T]` | Q 调制后的时域信号 |
| `y_time` | `[B, 1, 1, T]` | 经过信道的接收信号 |
| `r_freq` | `[B, 1, 1, M, N]` | Q 解调后的频域符号 |
| `llr` | `[B, 1, 1, K]` | 对数似然比 |

其中 `B=batch_size`，`N=FFT_SIZE=32`，`M=NUM_OFDM_SYMBOL=3`。

---

## 7. 配置文件参数一览 (`config.py`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `CARRIER_FREQ` | 3.5 GHz | 载波频率 |
| `DELAY_SPREAD` | 600 ns | TDL 固定模式延迟扩展 |
| `FFT_SIZE` | 32 | FFT 点数（子载波数） |
| `CYCLIC_PRFX_LEN` | 16 | 循环前缀长度 |
| `NUM_BITS_PER_SYMBOL` | 4 | 每 QAM 符号比特数 (16QAM) |
| `TOT_SYMBOLS_TO_DELIVER` | 64 | 每帧有效 QAM 符号数 |
| `NUM_OFDM_SYMBOL` | 3 | 每帧 OFDM 符号数 |
| `OFDM_SYMBOLS_FOR_PILOT_INDICES` | [0] | 第 0 个 OFDM 符号用于导频 |
| `SYMBOL_RATE` | 1 MHz | 符号速率（决定带宽） |
| `SEED` | 42 | 随机种子 |

**信道模型** (`channel.py`)：当前使用 `tdl_randomDS`，即 TDL-A 模型，延迟扩展在 [10ns, 600ns] 范围内对每个 batch 随机采样。
