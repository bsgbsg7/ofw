# Main_OTFS_CNN — CNN 均衡器实验

## 目的

测试用 CNN 联合均衡器替代原始模型中的单抽头均衡器（`r_freq * q`），
观察是否能促进 Q-creator 学习 OTFS 类波形。

## 核心思路

原始接收链路：
```
y_time → Q^H 解调 → r_freq → 单抽头均衡(r_freq * q) → 逐符号解映射 → LLR
```

CNN 版本：
```
y_time → Q^H 解调 → r_freq → CNN 联合均衡器 → 逐符号解映射 → LLR
```

**保留 Q^H 解调和 Sionna Demapper**（已内建 16QAM 星座知识），
仅用 CNN 替代单抽头均衡器。CNN 对 2×32 时频网格做联合处理，
能够消除子载波间和 OFDM 符号间的干扰（这是 OTFS 所必需的）。

## 文件结构

```
Main_OTFS_CNN/
├── config.py                    ← 复制自 Main_OTFS/config.py
├── channel_tv.py                ← 复制自 Main_OTFS/channel_tv.py
├── utils/                       → 指向 Main_OTFS/utils/ 的符号链接
├── legends.py                   → 指向 Main_OTFS/legends.py 的符号链接
├── src/qQ_Method/
│   ├── qQ_Model_TV_CNN.py       ← 新模型：CNN 均衡器 + Demapper
│   ├── qQ_creator_layer.py      → 符号链接（Q-creator）
│   ├── Q_Modulator.py           → 符号链接
│   ├── Q_Demodulator.py         → 符号链接
│   └── qQ_uncertainty_model.py  → 符号链接
├── transfer_weights.py          ← 权重迁移脚本（从预训练的 Q-creator）
├── train_final_cnn.py           ← 训练脚本
└── visualize_Q_waveforms.py     ← Q 矩阵可视化
```

## 关键设计

### CNNEqualizer

小型的残差 CNN，以 Q 解调后的 2×32 时频网格作为输入：

```
输入：(B, 2, 32) 复数
  → 分离实部/虚部 → (B, 2, 32, 2)
  → 3 个 Conv2D(32, 2×3) 块
  → Conv2D(2, 2×3) → 残差校正
  → 输出 = 输入 + 校正（残差连接）
  → 送入 Sionna Demapper
```

### 权重迁移

`transfer_weights.py` 将预训练的 Q-creator + 不确定性模型权重
从 `Main_OTFS/weights-qQ_Method_TV` 迁移到 CNN 模型，
同时保持 CNN 均衡器为随机初始化。

## 使用方式

```bash
# 1. 迁移预训练权重
python transfer_weights.py

# 2. 训练
python train_final_cnn.py

# 3. 可视化 Q 波形
python visualize_Q_waveforms.py
```

## 状态

**实验性** — CNN 均衡器训练仍在调试中。
初始预训练 Q-creator + 随机 CNN + Demapper 在 20dB 下的 BER 为 ~0.35，
但端到端微调不稳定。可能的下一步：
- 冻结 Q-creator，先单独训练 CNN 均衡器
- 更好地初始化 CNN 权重
- 使用更简单的均衡器架构（例如，小型 MLP）
