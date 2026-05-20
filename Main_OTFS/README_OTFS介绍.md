# DeepOFW-OTFS：基于时变信道的自适应波形学习

## 1. 项目动机

原始DeepOFW模型使用**静态TDL信道**（速度=0，仅改变延迟扩展），网络学会了：
- 低延迟扩展 → TDM波形（时分复用，低PAPR）
- 高延迟扩展 → OFDM波形（频分复用，抗多径）

但现实中高速移动场景（高铁、V2X）下，信道具有**时变特性**（多普勒效应）。OFDM在高多普勒下性能严重退化，而OTFS（正交时频空间调制）通过在时延-多普勒域操作，天然适合时变信道。

**本项目目标**：让网络在时变信道下自适应学习波形，期望：
- 低多普勒（低速）→ 学出OFDM-like波形
- 高多普勒（高速）→ 学出OTFS-like波形（时频同时扩展）

## 2. 与原始DeepOFW的关键区别

| 对比项 | 原始DeepOFW (Main/) | OTFS变体 (Main_OTFS/) |
|--------|---------------------|----------------------|
| 信道类型 | TDL-A，速度=0（静态） | TDL-A，速度=[3,120] m/s（时变） |
| 网络输入 | 单次CIR快照 `(batch, l_max, 1)` | 多次CIR快照 `(batch, l_max, 3)` |
| 条件变量 | 延迟扩展（DS） | 延迟扩展 + 多普勒频移 |
| 期望输出 | TDM ↔ OFDM 自适应 | OFDM ↔ OTFS 自适应 |
| 延迟扩展 | [10, 600] ns | [50, 300] ns |
| 速度范围 | 0 m/s | [3, 120] m/s（约10~432 km/h） |

## 3. 核心设计

### 3.1 时变信道模型

```python
# channel_tv.py
channel_model = TDL_RandomDS(
    model="A",
    delay_spread_min=50e-9,
    delay_spread_max=300e-9,
    carrier_frequency=3.5e9,
    min_speed=3.0,      # 最低速度 3 m/s ≈ 10 km/h
    max_speed=120.0     # 最高速度 120 m/s ≈ 432 km/h
)
```

Sionna的TDL模型使用sum-of-sinusoids方法生成信道系数，当speed>0时，信道路径系数随时间变化：

$$h(t, \tau) = \sum_i \alpha_i(t) \cdot \delta(\tau - \tau_i)$$

其中路径系数 $\alpha_i(t)$ 随时间变化，变化速率取决于最大多普勒频移 $f_d = v \cdot f_c / c$。

### 3.2 多快照CIR输入（核心创新）

原模型只观察一次信道冲激响应，无法感知信道的时间变化。新模型在**每个OFDM符号边界**提取一次CIR：

```
OFDM符号结构：[CP | 数据] [CP | 数据] [CP | 数据]
                ↑ CIR快照0    ↑ CIR快照1    ↑ CIR快照2

网络输入: (batch, 9个时延tap, 3个时间快照)
```

这样网络看到的是一个**二维时延-时间矩阵**：
- 行方向（时延维度）→ 多径信息
- 列方向（时间维度）→ 多普勒信息

高多普勒 → 列间变化大（CIR在不同符号间差异大）
低多普勒 → 列间变化小（CIR近似不变）

### 3.3 网络架构

使用与原模型相同的 `qQ_creator_conv_gru` 架构：
- Complex Conv1D → 提取时延-时间的局部特征
- Complex GRU → 捕获序列相关性
- Dense → 输出Q矩阵 (32×32) 和检测向量q (32)

**无需修改架构**，因为Conv1D本身支持多通道输入：
- 原模型：channels=1（单快照）
- 新模型：channels=3（三个时间快照）

### 3.4 损失函数

与原模型完全相同：
$$L = e^{-\sigma_R} \cdot \text{BCE} + e^{-\sigma_P} \cdot \text{PAR} + e^{-\sigma_\Theta} \cdot \varepsilon + \sigma_R + \sigma_P + \sigma_\Theta$$

不确定性网络根据信道特征（延迟扩展）自适应调整各项权重。

## 4. 预期行为

训练收敛后，期望观察到：

### 低多普勒（低速，~3 m/s）
- Q矩阵应趋向IDFT矩阵（标准OFDM）
- 波形在频域呈窄带子载波
- 因为静态信道下OFDM已经是最优方案

### 高多普勒（高速，~120 m/s）
- Q矩阵应学出**时频联合扩展**的结构
- 波形在时域和频域都有能量分布
- 类似OTFS的延迟-多普勒域调制特性
- 每个符号的能量分散在多个时频格点，抵抗多普勒造成的子载波间干扰（ICI）

## 5. 文件结构

```
DeepOFW/Main_OTFS/
├── config.py                  # 系统参数（含SPEED_MIN/MAX）
├── channel_tv.py              # 时变信道（TDL + 随机速度）
├── legends.py → ../Main/      # 符号链接
├── utils → ../Main/utils/     # 符号链接（共用工具函数）
├── src/
│   └── qQ_Method/
│       ├── qQ_Model_TV.py     # ★ 核心模型（多快照CIR输入）
│       ├── qQ_creator_layer.py  # 网络层（Conv-GRU，复制自Main）
│       ├── Q_Modulator.py       # Q调制器（复制自Main）
│       ├── Q_Demodulator.py     # Q解调器（复制自Main）
│       └── qQ_uncertainty_model.py  # 不确定性模型（复制自Main）
├── train_stage1_otfs.py       # 阶段1：从零训练
├── train_stage2_otfs.py       # 阶段2：高SNR微调
├── train_stage3_otfs.py       # 阶段3：扩展训练
└── run_full_pipeline.sh       # 一键运行脚本
```

## 6. 训练方法

### 环境准备
```bash
source activate deepofw
CONDA_PREFIX=$(conda info --base)/envs/deepofw
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

### 一键训练
```bash
cd /home/v-haoliu3/EfficientLLM/ShiZheng/DeepOFW/Main_OTFS
bash run_full_pipeline.sh
```

### 分步训练
| 阶段 | 脚本 | 学习率 | SNR范围 | 说明 |
|------|------|--------|---------|------|
| Stage 1 | `train_stage1_otfs.py` | 0.001 | [10, 25] dB | 从零训练，保存`weights-qQ_Method_TV_initial` |
| Stage 2 | `train_stage2_otfs.py` | 0.0001 | [20, 25] dB | 加载Stage 1微调 |
| Stage 3 | `train_stage3_otfs.py` | 0.0005 | [10, 25] dB | 扩展训练，保存`weights-qQ_Method_TV` |

## 7. 验证方法

训练完成后，可通过以下方式验证OTFS-like行为：

1. **波形可视化**：扫描不同速度（3~120 m/s），观察Q矩阵的时域/频域结构
2. **BER vs SNR**：在不同多普勒条件下对比BER曲线
3. **CCDF对比**：对比不同速度下的PAPR分布
4. **时频网格可视化**：将Q矩阵变换到时延-多普勒域，观察能量分布

## 8. 理论依据

OTFS相比OFDM的优势在于高多普勒场景：
- OFDM在频域正交，多普勒导致子载波间干扰（ICI）
- OTFS在时延-多普勒域操作，将时变信道转化为2D卷积，分集增益更大

DeepOFW框架的灵活性允许网络自动发现最优波形：
- 如果信道静态 → OFDM已经最优，网络不需要改变
- 如果信道时变 → 网络需要找到一种能同时利用时间和频率分集的波形
- 这种波形理论上应类似OTFS的时频联合扩展特性
