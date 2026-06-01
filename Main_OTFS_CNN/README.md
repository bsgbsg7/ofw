# Main_OTFS_CNN — MLP 联合均衡器实验

## 目的

测试用 MLP 联合均衡器替代原始模型中的单抽头均衡器（`r_freq * q`），
观察是否能促进 Q-creator 学习 OTFS 类波形。

## 核心思路

原始接收链路：
```
y_time → Q^H 解调 → r_freq → 单抽头均衡(r_freq * q) → 逐符号解映射 → LLR
```

MLP 版本：
```
y_time → Q^H 解调 → r_freq → q 缩放 → MLP 联合均衡器 → 逐符号解映射 → LLR
```

**保留 q 缩放 + Q^H 解调 + Sionna Demapper**（已内建 16QAM 星座知识），
仅用 MLP 替代单抽头均衡器中的逐子载波独立处理。
MLP 对所有 64 个 QAM 符号做联合处理，能够消除子载波间和 OFDM 符号间的干扰。

## 文件结构

```
Main_OTFS_CNN/
├── config.py                    ← 复制自 Main_OTFS/config.py
├── channel_tv.py                ← 复制自 Main_OTFS/channel_tv.py
├── utils/                       → 指向 Main_OTFS/utils/ 的符号链接
├── legends.py                   → 指向 Main_OTFS/legends.py 的符号链接
├── src/qQ_Method/
│   ├── qQ_Model_TV_CNN.py       ← 新模型：MLP 均衡器 + q 缩放 + Demapper
│   ├── qQ_creator_layer.py      → 符号链接（Q-creator）
│   ├── Q_Modulator.py           → 符号链接
│   ├── Q_Demodulator.py         → 符号链接
│   └── qQ_uncertainty_model.py  → 符号链接
├── transfer_weights.py          ← 权重迁移脚本（从预训练的 Q-creator）
├── train_final_cnn.py           ← 两阶段训练脚本
├── visualize_Q_waveforms.py     ← Q 矩阵可视化
├── Q_waveform_inspection.png    ← Q 波形对比图
└── cir_input_comparison.png     ← CIR 输入对比图
```

## 关键设计

### MLPEqualizer

轻量级 MLP，零初始化残差连接：

```
输入：(B, 128) 实数 — 64 个 QAM 符号展平（经过 q 缩放）
  → Dense(256) + BN + ReLU
  → Dense(256) + BN + ReLU
  → Dense(128) 零初始化（kernel_initializer='zeros'）
  → 输出 = 输入 + 校正（残差连接）
  → 送入 Sionna Demapper
```

参数量约 **132K**，非常轻量。

### 关键修复：q 向量缩放

在 MLP 之前，必须应用 Q-creator 的 `q` 向量进行逐子载波缩放。
这是预训练 Q-creator 输出的一部分，没有它会导致符号严重失真。
加上 `q` 后，零初始化 MLP（输出=输入）的 BER@20dB 从 0.35 降至 **0.088**。

### 两阶段训练

| 阶段 | 迭代次数 | Q-creator | 损失函数 | SNR 范围 |
|------|---------|-----------|---------|---------|
| Phase 1 | 1500 | **冻结** | MSE（MLP 输出 vs 真实 QAM 符号） | [10, 25] dB |
| Phase 2 | 2000 | 联合微调 | BCE（通过 Demapper） | [0, 25] dB |

- Phase 1 使用 MSE 监督学习 —— 平滑凸损失，快速收敛
- Phase 2 使用 BCE 端到端 —— 针对 BER 优化

## 训练结果

| 指标 | 初始值 | Phase 1 最佳 | Phase 2 最佳 |
|------|--------|-------------|-------------|
| BER@20dB | 0.088 | 0.033 | **0.0225** |
| MSE | 0.083 | 0.066 | — |
| 迭代次数 | 0 | 1100 | 1500 (P2) |

**结论：MLP 联合均衡器成功学习到有用的均衡校正，BER 从 0.088 降至 0.0225。**

与原始模型对比：
- 原始 one-tap 均衡器 + Demapper：BER@20dB = **0.0139**（训练 13000 次迭代）
- MLP 联合均衡器 + Demapper：BER@20dB = **0.0225**（训练 3500 次迭代）

MLP 版本仅用 1/4 的训练迭代次数达到了接近原始模型的性能，
证明联合均衡方法有效。更多迭代可能进一步缩小差距。

### Q 矩阵分析

可视化显示 Q 矩阵在三种场景下的距离：

| 场景 | d_IDFT | d_I | 倾向 |
|------|--------|-----|------|
| 平坦 3m/s | 1.411 | 1.015 | 略近 TDM |
| 多径 3m/s | 1.391 | 1.014 | 略近 TDM |
| 多径 120m/s | 1.390 | 1.015 | 略近 TDM |

Q 矩阵在不同场景间分化不明显 —— Q-creator 仅在 Phase 2 中微调了 2000 步，
不足以学会 OTFS 类波形。更长时间的训练或更强的多普勒信号（更高速度）
可能是必需的。

## 使用方式

```bash
# 1. 迁移预训练权重
python transfer_weights.py

# 2. 训练（两阶段）
python train_final_cnn.py

# 3. 可视化 Q 波形
python visualize_Q_waveforms.py
```

## 调试历程与经验教训

1. ❌ **CNN 从零开始 + 原始时域信号** → BER 停留在 0.5（CNN 必须隐式学习 Q^H）
2. ❌ **CNN + Q^H + 无 Demapper** → BER 停留在 0.5（CNN 必须学习 16QAM 星座图）
3. ❌ **CNN + Q^H + Demapper + 无 q 缩放** → BER ~0.35 但不收敛（符号缩放错误）
4. ❌ **MLP + Q^H + Demapper + 无 q 缩放** → BER 0.35→0.50，MSE 在下降但 BER 在上升
5. ✅ **MLP + Q^H + q 缩放 + Demapper + 两阶段训练** → BER 0.088→0.0225

**关键教训**：
- 始终保留 Sionna Demapper（它了解 16QAM 星座图）
- 始终应用 q 向量缩放（每个子载波需要正确的幅度/相位）
- 两阶段训练（先冻结 Q-creator 用 MSE 训练 MLP，再联合微调）对稳定性至关重要
- 零初始化的残差连接确保训练从"与原始相同"开始
