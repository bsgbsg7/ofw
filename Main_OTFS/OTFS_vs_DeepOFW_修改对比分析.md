# OTFS 训练模型 vs DeepOFW 原始模型 — 修改对比分析

> 对比基准：`/home/shizheng/baseline/ofw-autodl/Main/` (DeepOFW 原始代码)  
> 修改版本：`/home/shizheng/baseline/ofw-hao/Main_OTFS/` (OTFS 时变信道版本)

---

## 一、核心设计理念变化

| 维度 | DeepOFW 原始 | OTFS 修改版 |
|------|-------------|------------|
| 信道类型 | **静态** TDL（速度=0） | **时变** TDL（速度 0.5~120 m/s） |
| 学习目标 | 平坦→TDM，多径→OFDM | 平坦→TDM，多径低速→OFDM，多径高速→OTFS |
| 网络感知能力 | 仅感知多径结构 | 同时感知多径结构 + 多普勒（时变）特征 |
| 输入信息量 | 单个 CIR 快照 | 12 个时间维度 CIR 快照 |

---

## 二、config.py 参数对比

```diff
# ============ 新增参数 (OTFS 独有) ============
+ SPEED_MIN = 0.5              # 最低速度 [m/s]，接近静止行人
+ SPEED_MAX = 120.0            # 最高速度 [m/s]，~432 km/h 高铁
+ DELAY_SPREAD_MIN = 10e-9     # 最小时延扩展，近平坦衰落
+ DELAY_SPREAD_MAX = 600e-9    # 最大时延扩展，丰富多径
+ NUM_TIME_SNAPSHOTS = 12      # 时间快照数！（核心新增）
+ PAPR_WEIGHT = 5.0            # PAPR 损失权重
+ PAPR_THRESHOLD_DB = 0.0      # PAPR 阈值

# ============ 修改的参数 ============
- DELAY_SPREAD = 600e-9         # 原始：固定值
+ DELAY_SPREAD = 100e-9         # OTFS：变为后备值（实际使用 DELAY_SPREAD_MIN/MAX 范围）
```

**关键参数 `NUM_TIME_SNAPSHOTS = 12`**：  
- 原始 DeepOFW：Q-creator 输入形状 = `(batch, l_tot, 1)` — 仅 1 个 CIR 快照
- OTFS 修改版：Q-creator 输入形状 = `(batch, l_tot, 12)` — 12 个时间维度的 CIR 快照
- 采样策略：在整个时域轴上均匀采样，捕获信道的时间演化/多普勒信息

---

## 三、信道模型对比 (channel.py vs channel_tv.py)

| 维度 | DeepOFW 原始 (`channel.py`) | OTFS 修改版 (`channel_tv.py`) |
|------|---------------------------|------------------------------|
| 信道类 | `TDL_RandomDS` | `TDL_RandomDS` (相同) |
| TDL 模型 | TDL-A (23 径) | TDL-A (23 径) |
| 时延扩展 | 10~600 ns 随机 | 10~600 ns 随机 |
| **速度** | **`min_speed=0.0, max_speed=0.0`** | **`min_speed=0.5, max_speed=120.0`** |
| 多普勒效应 | 无 | 有，每样本随机 |
| 对训练的影响 | 信道仅频率选择性 | 信道兼有频率选择性 + 时间选择性 |

```python
# DeepOFW 原始 (channel.py):
tdl_randomDS = TDL_RandomDS(
    model="A",
    delay_spread_min=10e-9,
    delay_spread_max=600e-9,
    min_speed=0.0,    # ← 静态
    max_speed=0.0     # ← 静态
)

# OTFS 修改版 (channel_tv.py):
channel_model = TDL_RandomDS(
    model="A",
    delay_spread_min=10e-9,
    delay_spread_max=600e-9,
    min_speed=0.5,    # ← 接近静止
    max_speed=120.0   # ← 高铁级别
)
```

---

## 四、模型架构对比 (qQ_Model.py vs qQ_Model_TV.py)

### 4.1 CIR 提取方式

**这是最核心的修改。** 原始 DeepOFW 通过 delta 脉冲探测获取 CIR，然后只取 1 个快照：

```python
# DeepOFW 原始: 单快照 (batch, l_max, 1)
pilots_post_channel = tf.expand_dims(y_time[:,0,0,:self._l_max], axis=-1)
```

```python
# OTFS 修改版: 多快照均匀采样 (batch, l_tot, NUM_TIME_SNAPSHOTS=12)
h_2d = h_time[:, 0, 0, 0, 0, :, :]           # (batch, num_time_steps, l_tot)
total_time = tf.shape(h_2d)[1]
stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
pilots_post_channel = tf.gather(h_2d, indices, axis=1)  # (batch, 12, l_tot)
pilots_post_channel = tf.transpose(pilots_post_channel, [0, 2, 1])  # (batch, l_tot, 12)
```

### 4.2 信道特征提取

| 维度 | DeepOFW 原始 | OTFS 修改版 |
|------|-------------|------------|
| 特征维度 | **1** (仅 RMS delay spread) | **5** (rms_ds + doppler_ind + n_taps + 2 个 log 变体) |
| 输入不确定性网络 | `rms_ds` — shape `(batch, 1)` | `channel_features` — shape `(batch, 5)` |
| 多普勒感知 | 无 | `doppler_ind` = CIR 沿时间维度的方差 |
| 多径丰富度 | 无 | `n_taps` = 超过 10% 最大功率的抽头数 |

```python
# OTFS 新增的 5 维信道特征:
channel_features = tf.stack([
    rms_ds * 1e9,               # RMS 延迟扩展，缩放到 ns
    doppler_ind * 1e3,          # 多普勒指标，缩放到 1e-3
    n_taps,                     # 显著多径抽头数
    tf.math.log(rms_ds * 1e9 + 1e-10),   # log 变换
    tf.math.log(doppler_ind * 1e3 + 1e-10)  # log 变换
], axis=-1)  # shape: (batch, 5)
```

### 4.3 不确定性网络裁剪范围

| 参数 | DeepOFW 原始 | OTFS 修改版 |
|------|-------------|------------|
| `UncertaintyModel_2D` `min_log_sigma` | **-10.0** | **-3.0** |
| `UncertaintyModel_2D` `max_log_sigma` | **+10.0** | **+3.0** |
| 权重范围 `exp(-log_sigma)` | [4.5e-5, 22026] (极端) | [0.05, 20] (适中) |

OTFS 收紧不确定性权重的范围，防止 BCE/PAPR 权重塌缩到极端值。

### 4.4 模型类名

| DeepOFW 原始 | OTFS 修改版 |
|-------------|------------|
| `qQ_MODEL` | `qQ_MODEL_TV` |

---

## 五、Q-Creator 架构对比 (qQ_creator_layer.py)

### 5.1 原始 DeepOFW 架构 (单路径)

```
输入: (batch, l_tot, 1) complex
  │
  ├─ Conv1D (64 filters, kernel=32) — Complex
  ├─ GRU (1024 units) — Complex, 沿 delay 维度
  ├─ Feedforward (ff_dim→gru_units) + Residual + Dropout
  ├─ Global Mean Pooling
  └─ Output: Q (N×N) + q (N)
```

**特点**：仅在 delay（多径）维度上做处理，没有时间维度建模。

### 5.2 OTFS 修改版架构 (双路径)

```
输入: (batch, l_tot, 12) complex
  │
  ├─ TimePositionalEncoding (PE) — 注入时间位置信息
  │
  ├─ Path A (Delay-dim):  Conv1D(64,32) → ComplexGRU(1024) — 多径结构
  │
  ├─ Path B (Time-dim):   Transpose → Concat(real+imag) → Conv1D(64,16)
  │                        → Conv1D(64,3, cross-snapshot) → Mean Pool — 多普勒特征
  │
  ├─ Fusion: Concat(Path A + Path B broadcasted) → Complex Dense(gru_units)
  │
  ├─ Feedforward (ff_dim→gru_units) + Residual + Dropout
  ├─ Global Mean Pooling
  └─ Output: Q (N×N) + q (N)
```

**新增关键组件**：

| 组件 | 功能 | 参数 |
|------|------|------|
| `TimePositionalEncoding` | 为 12 个快照注入正弦位置编码 | `pe_scale=0.10` |
| Path B Conv1D (delay-feature) | 沿 delay 维度提取特征 | 64 filters, kernel=16 |
| Path B Conv1D (cross-time) | 建模相邻时间快照间的交互 | 64 filters, kernel=3 |
| Fusion Dense | 融合双路径信息 | Complex Dense(gru_units) |

### 5.3 额外架构变体（仅 OTFS 版本有）

在 OTFS 的 `qQ_creator_layer.py` 中，除了主用的 `qQ_creator_conv_gru`（双路径），还实现了两个备选架构：

| 架构 | 描述 | 参数量 |
|------|------|--------|
| `qQ_creator_conv2d` | Conv2D (2 blocks, 3×3, 32→64) + 4 统计量结构化池化 | ~5M |
| `qQ_creator_conv2d_v2` | Conv2D (4 blocks, 3×3, 32→256) + 3 路径结构化池化 | ~9.5M |

---

## 六、训练脚本对比

### 6.1 学习率调度

| 维度 | DeepOFW 原始 | OTFS 修改版 |
|------|-------------|------------|
| 调度策略 | **固定 LR** (Adam 默认) | **Warmup + Cosine Decay** |
| Stage 1 LR | `0.0005` (固定) | `0.001 → 0.0001` (cosine decay) |
| Stage 2 LR | `0.0001` (固定) | `1e-4 → 1e-5` (cosine decay) |
| Stage 3 LR | `0.0001` (固定) | `5e-4 → 1e-6` (cosine decay) |
| Warmup | 无 | Stage1: 200步, Stage2: 100步, Stage3: 200步 |
| 梯度裁剪 | **无** | **Global Norm Clip = 5.0** |

### 6.2 训练超参数

| 维度 | DeepOFW 原始 | OTFS Stage 1 | OTFS Stage 2 | OTFS Stage 3 |
|------|-------------|-------------|-------------|-------------|
| 迭代次数 | 5000 / 4000 / 20000 | 5000 | 3000 | 10000 |
| 有效 batch | 2560 | 2560 | 2560 | 2560 |
| SNR 训练范围 | [10, 25] dB (S1) / [20, 25] (S2) | [10, 25] dB | [10, 25] dB | [10, 25] dB |
| 评估 SNR | 20 dB / 24 dB (S3) | 20 dB | 20 dB | 20 dB |
| 评估 batch | 200 / 300 (S3) | 200 | 200 | 300 |
| 评估次数 | 5 | 5 | 5 | 5 |
| 早停耐心值 | 15 (S1/S2) / 20 (S3) | 15 | 15 | 20 |
| PAPR 损失 | 开启 | **关闭** | **关闭** | **关闭** |

### 6.3 训练循环差异

```python
# DeepOFW 原始: 返回 3 个值
total_loss, par, bce_loss = model_train(batch_size, ebno)

# OTFS 修改版: 返回 1 个值 (loss)
loss = model_train(batch_size, ebno)
```

### 6.4 新增功能：Q 矩阵快照保存

OTFS Stage 3 新增每 500 步保存 Q-creator 层权重的功能，用于训练进度分析：

```python
if i % 500 == 0 and i > 0:
    q_snapshot = model_train._qQ_creator_layer.get_weights()
    snapshot_file = f'Q_snapshot_stage3_iter_{i}.pkl'
    with open(snapshot_file, 'wb') as f:
        pickle.dump(q_snapshot, f)
```

### 6.5 损失函数返回差异

| 版本 | `call()` 训练模式下返回值 | 说明 |
|------|--------------------------|------|
| DeepOFW 原始 | `(total_loss, par_mean, bce_mean)` | 返回 3 个独立的 loss 分量 |
| OTFS 修改版 | `total_loss` | 仅返回总 loss，日志在模型内部打印 |

OTFS 版将训练日志（`tf.print`）移到了模型内部的 `call()` 方法中，简化了训练循环代码。

---

## 七、Q-Modulator / Q-Demodulator

**这两个文件在两个版本中完全相同。** 核心操作不变：

```python
# Modulation: x_time = x_freq @ Q  (替代 IFFT)
x_time = tf.einsum('bxyzi,bij->bxyzj', x_freq, Q)

# Demodulation: x_freq = x_time @ Q^H  (替代 FFT)
Q_herm = tf.linalg.adjoint(Q)
x = tf.einsum('bxyzi,bij->bxyzj', x, Q_herm)
```

---

## 八、文件结构对比

```
DeepOFW 原始 (Main/)                OTFS 修改版 (Main_OTFS/)
├── channel.py       (静态)      →  ├── channel_tv.py       (时变)
├── config.py        (基础)      →  ├── config.py           (+7 个新参数)
├── train_stage1.py  (固定 LR)   →  ├── train_stage1_otfs.py (cosine+clip)
├── train_stage2.py  (固定 LR)   →  ├── train_stage2_otfs.py (cosine+clip)
├── train_stage3.py  (固定 LR)   →  ├── train_stage3_otfs.py (cosine+clip+Q快照)
├── src/qQ_Method/
│   ├── qQ_Model.py   (单快照)   →  │   ├── qQ_Model_TV.py  (多快照+5特征)
│   ├── qQ_creator... (单路径)   →  │   ├── qQ_creator_layer.py (双路径+2D变体)
│   ├── qQ_uncertain...           →  │   ├── qQ_uncertainty_model.py (裁剪范围调整)
│   ├── Q_Modulator.py            →  │   ├── Q_Modulator.py (相同)
│   └── Q_Demodulator.py          →  │   └── Q_Demodulator.py (相同)
├── legends.py                    →  ├── legends.py → ../Main/legends.py (symlink)
├── utils/                        →  ├── utils/ → ../Main/utils/ (symlink)
                                    ├── train_final.py      (新增: 单阶段+PAPR)
                                    ├── train_v2.py         (新增: 结构化池化)
                                    └── train_v3.py         (新增: FiLM 调制)
```

---

## 九、总结：修改清单

| # | 修改项 | 原始 | 修改后 | 影响 |
|---|--------|------|--------|------|
| 1 | **信道类型** | 静态 (speed=0) | 时变 (speed 0.5~120 m/s) | 核心变更，使网络感知多普勒 |
| 2 | **CIR 快照数** | 1 个 | 12 个 (`NUM_TIME_SNAPSHOTS`) | 核心变更，提供时间维度信息 |
| 3 | **Q-creator 输入** | `(batch, l_tot, 1)` | `(batch, l_tot, 12)` | 输入通道 ×12 |
| 4 | **Q-creator 架构** | 单路径 (delay Conv1D+GRU) | 双路径 (delay+time) + PE | 新增时间/多普勒建模能力 |
| 5 | **位置编码** | 无 | `TimePositionalEncoding` (PE) | 注入时间位置信息 |
| 6 | **信道特征** | 1 维 (rms_ds) | 5 维 (rms_ds+doppler+n_taps+logs) | 不确定性网络获得更丰富信息 |
| 7 | **不确定性裁剪** | [-10, 10] | [-3, 3] | 防止权重塌缩 |
| 8 | **学习率调度** | 固定 Adam LR | Warmup + Cosine Decay | 更稳定的训练 |
| 9 | **梯度裁剪** | 无 | Global Norm Clip = 5.0 | 防止梯度爆炸 |
| 10 | **PAPR 损失** | Stage 1~3 开启 | Stage 1~3 关闭 | 优先优化 BER |
| 11 | **模型类名** | `qQ_MODEL` | `qQ_MODEL_TV` | TV = Time-Varying |
| 12 | **训练损失返回** | `(loss, par, bce)` 三元组 | `loss` 单值 | 简化训练循环 |
| 13 | **Q 快照保存** | 无 | Stage 3 每 500 步保存 | 训练进度可追溯 |
| 14 | **CIR 获取方式** | delta 脉冲探测 | 直接从 `h_time` 采样 | 更高效，且能获取多快照 |
| 15 | **备选架构** | 无 | Conv2D × 2 变体 | 实验灵活性 |

---

## 十、信号流程图对比

### DeepOFW 原始：
```
比特 → QAM → 资源网格 
  → [delta脉冲→信道→CIR(1快照)] → Q-creator → Q矩阵
  → Q调制 → 静态信道(h无时变) → Q解调 → 均衡 → 解映射 → LLR
  → BCE + PAPR(不确定性加权，仅用rms_ds)
```

### OTFS 修改版：
```
比特 → QAM → 资源网格
  → [时变信道(随机速度)] → 均匀采样12个CIR快照
  → [TimePositionalEncoding] → Q-creator(双路径: 多径+多普勒) → Q矩阵
  → Q调制 → 时变信道(h含多普勒) → Q解调 → 均衡 → 解映射 → LLR
  → BCE(不确定性加权，用5维信道特征: rms_ds+doppler+n_taps+logs)
```

---

> **文档生成时间**: 2026-06-01  
> **对比分支**: `DeepOFW (ofw-autodl/Main/)` vs `OTFS (ofw-hao/Main_OTFS/, branch: Hao)`  
> **分析工具**: Claude Code
