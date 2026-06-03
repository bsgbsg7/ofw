# OTFS Delay-Doppler 域验证报告

**模型**: `weights-qQ_Method_TV`  
**架构**: Conv1D + Complex GRU (1024) + Time-Path (双路: Delay + Doppler)  
**参数量**: ~11M  
**FFT_SIZE (N)**: 32  
**生成日期**: 2026-06-02  

---

## 0. 方法论与验证逻辑

### 现有验证的局限性

已有的验证脚本 (`verify_tdm_ofdm_otfs.py`, `otfs_proof_controlled.py`, `analyze_otfs_features.py`) 
只能回答一个问题: **Q 是不是 IDFT?** (即 Q ≠ OFDM)

| 已有方法 | 能回答什么 | 不能回答什么 |
|:---|:---|:---|
| `dist(Q, IDFT)` | Q 是不是 OFDM-like | Q 到底是不是 OTFS-like |
| `dist(Q, I)` | Q 是不是 TDM-like | — |
| 非对角能量占比 | Q 是否有扩展特性 | 扩展是否 = OTFS 的 DD 域结构 |
| Q^H·Q 正交性 | Q 是否能量守恒 | — |

**核心问题**: 这些指标只能做排除法 (排除 TDM, 排除 OFDM). 但不能正面证明 Q 学到的就是 OTFS.

### 本报告的验证方法 (来自 验证OTFS.md)

OTFS 的核心特征: 将 Delay-Doppler (DD) 域的每个符号通过 ISFFT + Heisenberg 变换, 
映射到时域波形。其逆过程 (时域波形 → SFFT → DD 域) 应产生一个 **尖锐的 2D 冲激**。

**验证逻辑链**:

1. **Q 列向量 = 波形基**: Q 的第 i 列 $\mathbf{q}_i$ 是当频率 bin i 输入为 1 时的时域波形
2. **逆向 DD 投影**: 对 $\mathbf{q}_i$ 做 FFT → 频域, reshape 为 TF 网格, 再做 SFFT → DD 域
3. **观察 DD 域模式**: 
   - 若是 **OTFS**: DD 域出现一个尖锐的 2D 冲激, 且不同列对应不同 DD 位置
   - 若是 **OFDM**: DD 域能量散布在多个位置 (因为 IDFT 不形成 DD 域冲激)
   - 若是 **TDM**: 完全不同的模式 (时域集中)
4. **亮点扫描测试**: 逐列做 DD 投影, 观察峰值位置是否在 2D 网格上规律移动

---

## 1. 基础 Q 矩阵指标

### 多场景对比

| 场景 | dist(Q, IDFT) | dist(Q, I) | Q^H·Q 非对角最大 | 判定 |
|:---|:---|:---|:---|:---|
| 平坦+低速 (TDM预期) | 1.5926 | 1.0246 | 0.4169 | ✅ 接近 TDM (I) |
| 多径+低速 (OFDM预期) | 1.8447 | 1.0380 | 0.6370 | ⚠️ 未完全靠拢 OFDM |
| 多径+中速 (过渡区) | 1.8616 | 1.0395 | 0.6555 | ⚠️ 未完全靠拢 OFDM |
| 多径+高速 (OTFS预期) | 1.7844 | 1.0348 | 0.6513 | ✅ 远离 TDM 和 OFDM → 可能 OTFS |
| 多径+超高速 | 1.7119 | 1.0311 | 0.5888 | ✅ 远离 TDM 和 OFDM → 可能 OTFS |

### 关键观察

- **平坦+低速**: dist_I=1.025, dist_IDFT=1.593 → 接近 TDM ✅
- **多径+低速**: dist_IDFT=1.845, dist_I=1.038 → 未靠拢 IDFT (可能的 OTFS 特性泄漏)
- **多径+高速**: dist_IDFT=1.784, dist_I=1.035 → 远离两者 ✅ → 可能 OTFS

---

## 2. Delay-Doppler 域逆向投影 (核心验证)

这是 **验证OTFS.md 中提出的最关键验证方法**。

### 方法说明

对 Q 矩阵的第 i 列 $\mathbf{q}_i$:
1. 构造频域输入: $\mathbf{x}_{freq} = \mathbf{e}_i$ (第 i 个位置为 1 的 one-hot 向量)
2. Q 调制: $\mathbf{x}_{time} = \mathbf{Q} \cdot \mathbf{x}_{freq}$
3. FFT 回频域: $\mathbf{y}_{freq} = \text{FFT}(\mathbf{x}_{time})$
4. 构造 2D TF 网格: reshape $\mathbf{y}_{freq}$ 为 $M \times N$ (频率 × 时间)
5. SFFT → DD 域: $\mathbf{Y}_{DD} = \text{SFFT}(\mathbf{Y}_{TF})$
6. 观察: DD 域是否出现 **尖锐的 2D 冲激**

### 各场景 DD 域分析结果

| 场景 | 中间列 DD 集中度 | 全列平均 DD 集中度 | 唯一峰值位置数 | 峰值覆盖率 | OTFS 特征? |
|:---|:---|:---|:---|:---|:---|
| 平坦+低速 (TDM预期) | 0.9966 | 0.9938 | 32/32 | 100.00% | ✅ 强 OTFS 特征 |
| 多径+低速 (OFDM预期) | 0.9611 | 0.9276 | 32/32 | 100.00% | ✅ 强 OTFS 特征 |
| 多径+中速 (过渡区) | 0.9501 | 0.9159 | 32/32 | 100.00% | ✅ 强 OTFS 特征 |
| 多径+高速 (OTFS预期) | 0.7494 | 0.6091 | 32/32 | 100.00% | 🔶 中等 OTFS 特征 |
| 多径+超高速 | 0.7298 | 0.5590 | 32/32 | 100.00% | 🔶 中等 OTFS 特征 |

### DD 域分析关键发现

- **高速 vs 低速 DD 集中度**: 高速=0.6091, 低速=0.9276
  → 高速下 DD 域能量 **反而更分散** — 这与 OTFS 预期相反。OTFS 应在 DD 域形成更集中的冲激。
- **峰值覆盖率**: 所有场景都是 100% (32/32)
  → 这是因为 Q 近似酉矩阵, 每列线性独立, 必然映射到不同的峰值位置。这不构成 OTFS 的正面证据。
- **⚠️ 方法论局限**: 当前 DD 网格仅为 32×2 (频率×时间), 2 个 Doppler bin 不足以形成有意义的 2D 冲激结构。要真正验证 OTFS, 需要更大的 DD 网格 (如 32×8 或更大)。

**修正解读**: 上述 DD 集中度数据说明每列 Q 的时域波形经过 FFT 后能量集中在特定频率 bin。高速下集中度降低意味着 Q 在高 Doppler 下将单个频率符号的能量**更广泛地扩散**到多个频率 bin。这种行为更接近 OFDM 在 Doppler 下的 ICI 效应而非 OTFS。

---

## 3. 正交性与特征值深度分析

OTFS 变换是酉变换 (Q^H·Q = I), 保证能量守恒和正交性。

### Q^H·Q 分析结果

| 场景 | 条件数 | 特征值范围 | 特征值 STD | 非对角最大 |
|:---|:---|:---|:---|:---|
| 平坦+低速 (TDM预期) | 1.334 | [0.557, 0.991] | 0.0705 | 0.4169 |
| 多径+低速 (OFDM预期) | 1.983 | [0.232, 0.910] | 0.1270 | 0.6370 |
| 多径+中速 (过渡区) | 2.116 | [0.184, 0.823] | 0.1231 | 0.6555 |
| 多径+高速 (OTFS预期) | 4.883 | [0.049, 1.159] | 0.2297 | 0.6513 |
| 多径+超高速 | 2.920 | [0.164, 1.400] | 0.2292 | 0.5888 |

### 正交性判定

- 平坦+低速 (TDM预期): ✅ 近似酉矩阵 (cond=1.33)
- 多径+低速 (OFDM预期): 🔶 部分满足酉性 (cond=1.98)
- 多径+中速 (过渡区): 🔶 部分满足酉性 (cond=2.12)
- 多径+高速 (OTFS预期): 🔶 部分满足酉性 (cond=4.88)
- 多径+超高速: 🔶 部分满足酉性 (cond=2.92)

---

## 4. CIR → Q 敏感度 2D 分析

扫描速度 × 延迟扩展参数空间, 观察 Q 在何时偏离 IDFT (即学到非 OFDM 变换)。

### dist(Q, IDFT) 热力图解读

- **速度范围**: [3, 10, 30, 60, 90, 120, 200] m/s
- **延迟扩展范围**: ['10ns', '50ns', '100ns', '200ns', '300ns', '500ns']

**理论预期**:
- 低速 & 低延迟 → 接近 I (TDM)
- 低速 & 高延迟 → 接近 IDFT (OFDM)
- 高速 & 高延迟 → 远离 IDFT (OTFS)

**实测结果**:
  - 10ns @ 3m/s: dist_IDFT=1.64 → 强非 OFDM (可能 OTFS)
  - 10ns @ 10m/s: dist_IDFT=1.62 → 强非 OFDM (可能 OTFS)
  - 10ns @ 30m/s: dist_IDFT=1.87 → 强非 OFDM (可能 OTFS)
  - 10ns @ 60m/s: dist_IDFT=1.51 → 强非 OFDM (可能 OTFS)
  - 10ns @ 90m/s: dist_IDFT=1.54 → 强非 OFDM (可能 OTFS)
  - 10ns @ 120m/s: dist_IDFT=1.67 → 强非 OFDM (可能 OTFS)
  - 10ns @ 200m/s: dist_IDFT=1.71 → 强非 OFDM (可能 OTFS)
  - 50ns @ 3m/s: dist_IDFT=1.57 → 强非 OFDM (可能 OTFS)
  - 50ns @ 10m/s: dist_IDFT=1.66 → 强非 OFDM (可能 OTFS)
  - 50ns @ 30m/s: dist_IDFT=1.58 → 强非 OFDM (可能 OTFS)
  - 50ns @ 60m/s: dist_IDFT=1.54 → 强非 OFDM (可能 OTFS)
  - 50ns @ 90m/s: dist_IDFT=1.63 → 强非 OFDM (可能 OTFS)
  - 50ns @ 120m/s: dist_IDFT=1.57 → 强非 OFDM (可能 OTFS)
  - 50ns @ 200m/s: dist_IDFT=1.50 → 强非 OFDM (可能 OTFS)
  - 100ns @ 3m/s: dist_IDFT=1.65 → 强非 OFDM (可能 OTFS)
  - 100ns @ 10m/s: dist_IDFT=1.60 → 强非 OFDM (可能 OTFS)
  - 100ns @ 30m/s: dist_IDFT=1.64 → 强非 OFDM (可能 OTFS)
  - 100ns @ 60m/s: dist_IDFT=1.85 → 强非 OFDM (可能 OTFS)
  - 100ns @ 90m/s: dist_IDFT=1.80 → 强非 OFDM (可能 OTFS)
  - 100ns @ 120m/s: dist_IDFT=1.77 → 强非 OFDM (可能 OTFS)
  - 100ns @ 200m/s: dist_IDFT=1.50 → 强非 OFDM (可能 OTFS)
  - 200ns @ 3m/s: dist_IDFT=1.74 → 强非 OFDM (可能 OTFS)
  - 200ns @ 10m/s: dist_IDFT=1.68 → 强非 OFDM (可能 OTFS)
  - 200ns @ 30m/s: dist_IDFT=1.55 → 强非 OFDM (可能 OTFS)
  - 200ns @ 60m/s: dist_IDFT=1.78 → 强非 OFDM (可能 OTFS)
  - 200ns @ 90m/s: dist_IDFT=1.69 → 强非 OFDM (可能 OTFS)
  - 200ns @ 120m/s: dist_IDFT=1.69 → 强非 OFDM (可能 OTFS)
  - 200ns @ 200m/s: dist_IDFT=1.65 → 强非 OFDM (可能 OTFS)
  - 300ns @ 3m/s: dist_IDFT=1.77 → 强非 OFDM (可能 OTFS)
  - 300ns @ 10m/s: dist_IDFT=1.54 → 强非 OFDM (可能 OTFS)
  - 300ns @ 30m/s: dist_IDFT=1.76 → 强非 OFDM (可能 OTFS)
  - 300ns @ 60m/s: dist_IDFT=1.67 → 强非 OFDM (可能 OTFS)
  - 300ns @ 90m/s: dist_IDFT=1.70 → 强非 OFDM (可能 OTFS)
  - 300ns @ 120m/s: dist_IDFT=1.63 → 强非 OFDM (可能 OTFS)
  - 300ns @ 200m/s: dist_IDFT=1.65 → 强非 OFDM (可能 OTFS)
  - 500ns @ 3m/s: dist_IDFT=1.46 → 强非 OFDM (可能 OTFS)
  - 500ns @ 10m/s: dist_IDFT=1.65 → 强非 OFDM (可能 OTFS)
  - 500ns @ 30m/s: dist_IDFT=1.49 → 强非 OFDM (可能 OTFS)
  - 500ns @ 60m/s: dist_IDFT=1.64 → 强非 OFDM (可能 OTFS)
  - 500ns @ 90m/s: dist_IDFT=1.73 → 强非 OFDM (可能 OTFS)
  - 500ns @ 120m/s: dist_IDFT=1.55 → 强非 OFDM (可能 OTFS)
  - 500ns @ 200m/s: dist_IDFT=1.56 → 强非 OFDM (可能 OTFS)

---

## 5. 时频扩展谱分析

OTFS 将一个符号的能量均匀扩展到整个时频网格。IDFT (OFDM) 也有均匀扩展特性。
关键区分: OTFS 的扩展模式随 Doppler 变化, OFDM 的扩展模式固定。

| 场景 | 时域扩展宽度 [samples] | 频域扩展宽度 [bins] | 与 IDFT 均匀扩展差异 |
|:---|:---|:---|:---|
| 平坦+低速 (TDM预期) | 1.08 | 9.31 | 8.16 |
| 多径+低速 (OFDM预期) | 3.68 | 9.60 | 5.57 |
| 多径+中速 (过渡区) | 4.33 | 9.67 | 4.93 |
| 多径+高速 (OTFS预期) | 8.82 | 9.88 | 0.76 |
| 多径+超高速 | 9.15 | 9.94 | 0.71 |

  (IDFT 的均匀扩展参考值: 9.24 samples)

---

## 6. Q 行列相关结构

| 场景 | 列间平均相关 | 行间平均相关 |
|:---|:---|:---|
| 平坦+低速 (TDM预期) | 0.0179 | 0.0179 |
| 多径+低速 (OFDM预期) | 0.0479 | 0.0476 |
| 多径+中速 (过渡区) | 0.0463 | 0.0461 |
| 多径+高速 (OTFS预期) | 0.0776 | 0.0762 |
| 多径+超高速 | 0.0696 | 0.0672 |

**解读**: 列间/行间相关越低, Q 的基向量越独立, 越接近一组正交基 (OTFS/OFDM 特性)。

---

## 7. Doppler 分辨率分析

测试 Q 对不同速度的敏感度。相邻速度间的 Q 差异越大 → Doppler 分辨率越高。

| 速度区间 | Q 差异 (Frobenius) |
|:---|:---|
| 3 → 10 m/s | 2.6746 |
| 10 → 30 m/s | 1.8052 |
| 30 → 60 m/s | 1.1524 |
| 60 → 90 m/s | 0.7332 |
| 90 → 120 m/s | 0.6625 |
| 120 → 150 m/s | 1.6146 |
| 150 → 200 m/s | 1.7929 |

- 低速区平均差异: 1.8774
- 高速区平均差异: 1.2008
  → 高低速区的 Q 变化相似 → Doppler 分辨率均匀

---

## 8. 综合判定

### 8.1 逐项检查汇总

| 验证项 | 平坦+低速 | 多径+低速 | 多径+高速 | 说明 |
|:---|:---|:---|:---|:---|
| 类型匹配 | ✅ (TDM) | ❌ (应接近 IDFT) | ✅ (远离两者) | — |
| 近似酉矩阵 (cond<3) | ✅ | ✅ | ❌ | OTFS 需要酉性 |
| DD域集中度>0.5 | ✅ | ✅ | ✅ | 但高速应更高才对 |
| Q 随 Doppler 变化 | — | — | Δdist=-0.06 | 变化方向不对 |

### 8.2 诚实评估：模型到底学到了什么？

#### ✅ 确实学到的能力

1. **平坦信道 → TDM**: 无论速度高低, 平坦信道下 Q 始终接近单位阵 I (dist_I ≈ 1.02)。网络正确识别了"无频率选择性 → 不需要 OFDM/OTFS"。

2. **Q 随 CIR 条件变化**: Q 不是固定矩阵。不同 CIR 输入产生不同的 Q, 网络确实从 CIR 中提取了信息。

3. **时域扩展随 Doppler 增加**: 高速下时域扩展宽度 (8.82 samples) 远大于低速 (3.68 samples), 接近 IDFT 均匀扩展 (9.24 samples)。这是 OTFS-like 行为。

4. **BER 可用**: 训练后的模型在所有信道条件下都能通信 (BER < 0.05 @ 20dB)。

#### ❌ 未学到的能力 (关键缺陷)

1. **多径+低速 ≠ OFDM**: 这是最重要的失败。在信道存在频率选择性但 Doppler 低的情况下, OFDM 是最优方案, 但网络学到的 Q 始终远离 IDFT (dist_IDFT ≈ 1.7-1.9)。
   - **原因分析**: 训练时速度在 [3, 120] m/s 范围内随机, 网络可能没有充分的低速样本。同时, 损失函数中没有显式引导 Q 靠拢 IDFT 的项。

2. **高速 ≠ 更 OTFS**: 高速下的 DD 域集中度 (0.61) 反而低于低速 (0.93), 这与 OTFS 预期相反 (OTFS 应在 DD 域更集中)。

3. **"排除法"判定 OTFS 的逻辑缺陷**: 现有脚本用 "dist_IDFT > 0.5 且 dist_I > 0.5 → OTFS" 来判定。但**所有**多径条件下的 Q 都满足这个条件 (因为 Q 始终 ≈ I + 小幅扰动)。这导致"只要不是平坦信道, 就是 OTFS"的假阳性。

4. **正交性随 Doppler 退化**: 高速下条件数达 4.88 (理想酉: 1.0), Q^H·Q 偏离单位阵。真正的 OTFS 应始终保持酉性。

### 8.3 根本原因分析

| 问题 | 根本原因 |
|:---|:---|
| Q 不靠拢 IDFT | 训练时速度范围 [3, 120] m/s 使得网络优先学习通用策略, 而非低速→OFDM 的条件化策略 |
| DD 域集中度反转 | DD 网格太小 (仅 32×2), 无法形成有意义的 2D 结构; 当前 DD 分析有方法论局限 |
| 正交性退化 | 高速下 CIR 变化剧烈, 网络为适应多变的信道牺牲了精确的酉性 |
| 排除法假阳性 | 现有 "OTFS = 非 IDFT 且非 I" 定义过于宽松, 几乎所有非平凡矩阵都满足 |

### 8.4 最终结论

**模型学到了"平坦→TDM, 多径→非标准扩展变换"的策略, 但尚未实现真正的 OTFS (DD 域集中映射)。**

| 方面 | 评分 | 说明 |
|:---|:---|:---|
| TDM 识别 | ⭐⭐⭐⭐⭐ | 完美: 平坦信道 → Q ≈ I |
| OFDM 识别 | ⭐ | 失败: 多径+低速不靠拢 IDFT |
| OTFS 特征 | ⭐⭐⭐ | 部分: 时域扩展、CIR 自适应存在, 但 DD 域行为不符合预期 |
| 正交性保持 | ⭐⭐⭐ | 部分: 低速好, 高速退化 |
| 整体评估 | ⭐⭐⭐ | 有自适应性但方向不完全正确 |

**关键 gap**: 网络缺乏"多径+低速 → OFDM (IDFT)"的映射, 这意味着它没有学会根据 Doppler 水平在 OFDM 和 OTFS 之间切换 — 而这正是 OTFS 学习的核心目标。

---

## 9. 附加验证方法建议

### 9.1 已在本报告中实现的新方法

1. **DD 域逆向投影** (来自 验证OTFS.md): ✅ 已实现 — `verify_otfs_dd_domain.py`
2. **DD 域网格亮点扫描**: ✅ 已实现
3. **特征值分析 (条件数)**: ✅ 已实现
4. **CIR 敏感度 2D 扫描**: ✅ 已实现
5. **Doppler 分辨率分析**: ✅ 已实现
6. **时频扩展谱分析**: ✅ 已实现

### 9.2 当前方法的方法论局限

1. **DD 网格太小**: FFT_SIZE=32, 仅 2 个数据 OFDM 符号 → DD 网格为 32×2。如此小的网格无法展示有意义的 2D 冲激结构。
2. **Q 作用于频域而非 DD 域**: 模型输入是频域符号, Q 映射 frequency→time。要验证 OTFS, 需要完整的 DD→TF→time 链。
3. **排除法误差**: "dist_IDFT > 0.5 且 dist_I > 0.5 → OTFS" 的判定对几乎所有非平凡酉矩阵都成立。

### 9.3 可进一步实施的方法

1. **完整 DD 域往返测试**: 构造 DD 域输入 → ISFFT → Q 调制 → 信道 → Q^H 解调 → SFFT → DD 域, 计算输入-输出 DD 域相关性
2. **互信息分析**: 计算 I(X; Y|CIR, Q) 在不同 Doppler 下的变化
3. **DD 域等效信道矩阵**: OTFS 将时变信道转化为 DD 域的 2D 卷积, 检查等效信道是否呈现块循环结构
4. **BER vs 归一化 Doppler (f_d·T)**: OTFS 应对归一化 Doppler 不敏感
5. **与理想 OTFS ISFFT 矩阵的相似度**: 将学习的 Q 与理想 ISFFT + Heisenberg 复合变换做对齐

### 9.4 训练改进建议 (针对发现的问题)

| 优先级 | 改进方向 | 具体措施 | 预期效果 |
|:---|:---|:---|:---|
| 🔴 高 | 引导 OFDM 学习 | 加入辅助 loss: `λ·‖Q − IDFT‖²` 对低 Doppler 样本加权 | 多径+低速 → Q ≈ IDFT |
| 🔴 高 | 扩大 DD 网格 | 增加 FFT_SIZE (64/128) 和 NUM_OFDM_SYMBOL (8+) | 有意义的 2D DD 结构 |
| 🟡 中 | DD 域结构 loss | 对高 Doppler 样本: 最小化 DD 域能量熵 (促进集中) | Q 在 DD 域形成冲激 |
| 🟡 中 | 更多时间快照 | 增加 NUM_TIME_SNAPSHOTS (24→48) | 更精细的 Doppler 分辨率 |
| 🟢 低 | 正交性约束 | 加入 `‖Q^H·Q − I‖²` 正则化 | 高速下保持酉性 |
| 🟢 低 | 分阶段训练 | 先固定 Q=IDFT 训练 BER, 再对高 Doppler 放开 Q | 确保 OFDM 基线

---

## 10. 生成的可视化文件

### 本报告新生成 (DD 域验证)

| 文件名 | 内容 |
|:---|:---|
| `OTFS_dd_verification_main.png` | DD 域核心验证大图 (Q矩阵 + DD投影 + 正交性 + 各列DD集中度) |
| `OTFS_dd_grid_scan_high_speed.png` | 高速 120m/s 场景 — 32列 DD 域网格逐列扫描 (验证"亮点遍历") |
| `OTFS_dd_grid_scan_low_speed.png` | 低速 3m/s 场景 — DD 域网格扫描 (对比) |
| `OTFS_cir_sensitivity_2d.png` | CIR → Q 敏感度 2D 热力图 (速度×延迟扩展) |
| `OTFS_orthogonality_analysis.png` | 正交性深度分析 (条件数 + 特征值分布 + 行列相关性 + Q在TDM-OFDM-OTFS空间位置) |
| `OTFS_doppler_spread_analysis.png` | Doppler 分辨率 & 时频扩展分析 |

### 已有验证脚本生成

| 文件名 | 脚本 | 内容 |
|:---|:---|:---|
| `verify_tdm_ofdm_otfs.png` | `verify_tdm_ofdm_otfs.py` | 6 场景 Q 矩阵对比 (幅度 + 差值 + 单符号扩展) |
| `OTFS_proof_controlled.png` | `otfs_proof_controlled.py` | 控制变量 Q 矩阵对比 (固定DS, 变速) |
| `Q_waveform_inspection.png` | `visualize_Q_waveforms.py` | Q 时域基底 + CIR 输入可视化 |
| `cir_input_comparison.png` | `visualize_Q_waveforms.py` | CIR Delay Profile 对比 |
| `architecture_and_Q_analysis.png` | `visualize_architecture_and_Q.py` | 架构图 + Q 矩阵分析 |

### 脚本

| 文件名 | 功能 |
|:---|:---|
| `verify_otfs_dd_domain.py` | **新**: 本报告的 DD 域验证脚本 (7 项检查 + 6 张图) |

---

*报告生成时间: 2026-06-02 | 模型: weights-qQ_Method_TV | N=32 | FFT_SIZE=32*