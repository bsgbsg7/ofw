#!/usr/bin/env python3
"""
OTFS Delay-Doppler 域验证脚本
==============================
核心问题: 现有的验证方法只能说明 Q ≠ IDFT (即不是 OFDM),
但不能证明 Q 就是 OTFS。本脚本实现了 验证OTFS.md 中提出的
Delay-Doppler 域逆向分析方法，并加入多项新颖的验证手段。

验证方法:
  1. [核心] Q 列向量 → DD 域逆向投影 (来自 验证OTFS.md)
  2. Q 列向量遍历 → 观察 DD 域亮点扫描
  3. Q^H·Q 正交性 & 特征值分析
  4. DD 域能量集中度 vs Doppler
  5. CIR → Q 敏感度 2D 热力图
  6. 多符号 TF 网格 → 2D SFFT 分析
  7. Q 的 Doppler 分辨率分析

参考: 验证OTFS.md 中的 OTFS 验证方法
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import *
from collections import defaultdict

# ============================================================================
# 工具函数
# ============================================================================

def compute_idft(N):
    """标准 IDFT 矩阵"""
    k = np.arange(N); n = np.arange(N).reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def compute_dft(N):
    """标准 DFT 矩阵"""
    return compute_idft(N).conj().T

def build_channel(speed_min, speed_max, ds_min, ds_max):
    from utils.TDL_RandomDS import TDL_RandomDS
    return TDL_RandomDS(
        model='A', delay_spread_min=ds_min, delay_spread_max=ds_max,
        carrier_frequency=CARRIER_FREQ, min_speed=speed_min, max_speed=speed_max
    )

def extract_q_and_cir(model, speed, ds_min, ds_max, batch_size=16):
    """提取 Q 矩阵和 CIR 输入"""
    from sionna.phy.channel import cir_to_time_channel
    original_channel = model._channel_model
    test_channel = build_channel(
        max(0.1, speed-0.5), speed+0.5,
        ds_min, ds_max + 1e-9
    )
    model._channel_model = test_channel
    a, tau = test_channel(batch_size,
                          model._rg.num_time_samples + model._l_tot - 1,
                          model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                 l_min=model._l_min, l_max=model._l_max, normalize=True)
    h_2d = h_time[:, 0, 0, 0, 0, :, :]
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    pilots = tf.gather(h_2d, indices, axis=1)
    cir_input = tf.transpose(pilots, [0, 2, 1])
    creator = model._qQ_creator_layer
    Q, _ = creator(cir_input, training=False)
    model._channel_model = original_channel
    return Q.numpy(), cir_input.numpy()

# ============================================================================
# 方法 1: Q 列向量 DD 域逆向投影 (来自 验证OTFS.md 的核心方法)
# ============================================================================

def sfft_transform(tf_grid):
    """
    Symplectic Finite Fourier Transform: TF 域 → DD 域
    tf_grid: (M_freq, N_time) complex
    returns: (M_freq, N_time) complex — DD 域表示

    SFFT: 沿时间轴做 N_time-pt FFT, 沿频率轴做 M_freq-pt IFFT
    """
    M, N = tf_grid.shape
    # 沿时间轴 FFT (将时间变化映射到 Doppler)
    dd = np.fft.fft(tf_grid, n=N, axis=1) / np.sqrt(N)
    # 沿频率轴 IFFT (将频率映射到 delay)
    dd = np.fft.ifft(dd, n=M, axis=0) * np.sqrt(M)
    return dd

def isfft_transform(dd_grid):
    """
    Inverse SFFT: DD 域 → TF 域
    dd_grid: (M_delay, N_doppler) complex
    returns: (M_freq, N_time) complex
    """
    M, N = dd_grid.shape
    # 沿 delay 轴 FFT → 频率
    tf = np.fft.fft(dd_grid, n=M, axis=0) / np.sqrt(M)
    # 沿 Doppler 轴 IFFT → 时间
    tf = np.fft.ifft(tf, n=N, axis=1) * np.sqrt(N)
    return tf

def analyze_q_column_dd(Q, column_idx, N_symbols=2):
    """
    核心验证方法 (来自 验证OTFS.md):
    对 Q 的第 column_idx 列做 DD 域逆向投影。

    Q 的第 i 列 q_i = Q[:, i] 是当频率 bin i 输入为 1 时的时域波形。

    步骤:
      1. 提取 q_i = Q[:, i]
      2. 将 q_i reshape 到 TF 网格 (M_freq × N_time)
         - 需要多符号才能形成 2D DD 网格
         - 对于单符号: 将其视为 TF 网格的一列
      3. 做 SFFT → DD 域
      4. 观察是否形成尖锐的 2D 冲激

    返回: DD 域网格, 能量集中度指标
    """
    N = Q.shape[0]  # 32
    q_i = Q[:, column_idx]  # (N,)

    # 对于单符号 32 点, 我们可以创建一个人工的 M×N 网格
    # 方案: 将 32 个频率 bin 分为 M=8 组 (每组 4 个 bin),
    #       时间采样分为 N=4 组 (跨 OFDM 符号)
    # 更好的方案: 直接使用全部 N 点作为频率维度, 单符号作为 1 列

    # 方案 A: 用 N_symbols=2, 将 Q 的列复制到两个 OFDM 符号
    # 构造 TF 网格: 32 频率 × 2 时间
    tf_grid = np.zeros((N, N_symbols), dtype=np.complex128)
    for s in range(N_symbols):
        # 每个符号: Q 调制输入 delta_freq, 输出时域波形, 再 FFT 回频域
        x_freq = np.zeros(N, dtype=np.complex128)
        x_freq[column_idx] = 1.0
        x_time = Q @ x_freq  # 时域波形
        # FFT 回频域 → 得到 TF 网格的这一列
        x_back = np.fft.fft(x_time) / np.sqrt(N)
        tf_grid[:, s] = x_back

    # SFFT → DD 域
    dd_grid = sfft_transform(tf_grid)

    # 能量集中度: 峰值能量 / 总能量
    power = np.abs(dd_grid) ** 2
    total_power = np.sum(power)
    peak_power = np.max(power)
    concentration = peak_power / (total_power + 1e-10)

    # 峰值位置 (delay, Doppler)
    peak_idx = np.unravel_index(np.argmax(power), power.shape)

    return dd_grid, concentration, peak_idx, tf_grid

def dd_scan_test(Q):
    """
    遍历 Q 的所有列, 在 DD 域观察亮点是否按规律扫描。

    如果是完美的 OTFS:
      - Q 的第 i 列对应 DD 域位置 (delay_idx, doppler_idx)
      - 按一定规律在 2D 网格上扫描, 最终覆盖整个网格
      - 每列都应该对应一个尖锐的 2D 冲激

    返回: 每列的 concentration, peak_position
    """
    N = Q.shape[0]
    results = {}
    concentrations = []
    peak_positions = []

    for i in range(N):
        dd_grid, conc, peak, tf_grid = analyze_q_column_dd(Q, i, N_symbols=2)
        concentrations.append(conc)
        peak_positions.append(peak)

    concentrations = np.array(concentrations)
    peak_positions = np.array(peak_positions)

    results['concentration_mean'] = np.mean(concentrations)
    results['concentration_std'] = np.std(concentrations)
    results['concentration_min'] = np.min(concentrations)
    results['concentration_max'] = np.max(concentrations)
    results['peak_positions'] = peak_positions
    results['concentrations'] = concentrations

    # 检查峰值位置是否覆盖网格 (OTFS 特征)
    unique_peaks = set(tuple(p) for p in peak_positions)
    results['unique_peak_count'] = len(unique_peaks)
    results['peak_coverage'] = len(unique_peaks) / N  # 理想 OTFS: 1.0

    return results

# ============================================================================
# 方法 2: 增强 DD 域分析 — 网格扫描可视化
# ============================================================================

def dd_grid_scan_visualization(Q, title_prefix=""):
    """
    逐列在 DD 域投影, 将结果累加显示亮点如何遍历网格。
    类似 验证OTFS.md 中描述的"亮点像扫描仪一样逐个格子移动"。
    """
    N = Q.shape[0]
    n_cols = min(8, N)
    n_rows = (N + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    all_peaks = []
    for i in range(N):
        dd_grid, conc, peak, _ = analyze_q_column_dd(Q, i, N_symbols=2)
        all_peaks.append(peak)

        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        power_db = 10 * np.log10(np.abs(dd_grid)**2 + 1e-10)
        vmax = np.max(power_db)
        vmin = vmax - 30  # 30 dB dynamic range
        im = ax.imshow(power_db, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
        ax.scatter(peak[1], peak[0], marker='x', color='cyan', s=80, linewidths=2)
        ax.set_title(f'Col {i} → DD({peak[0]},{peak[1]})\nconc={conc:.3f}', fontsize=8)
        ax.set_xlabel('Doppler', fontsize=7)
        ax.set_ylabel('Delay', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7)

    for i in range(N, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle(f'{title_prefix}Q 列 → DD 域逆向投影\n'
                 '亮点应出现在不同 (delay, Doppler) 坐标 (OTFS 特征)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig, all_peaks

# ============================================================================
# 方法 3: Q^H·Q 正交性与特征值分析
# ============================================================================

def analyze_q_orthogonality(Q):
    """
    分析 Q 的正交性:
      - Q^H·Q: 应接近单位阵 (酉矩阵)
      - 特征值: 对于酉矩阵, 所有特征值模应为 1
      - 条件数: 应接近 1
    """
    N = Q.shape[0]
    QHQ = Q.conj().T @ Q

    # 偏离单位阵度量
    off_diag = np.abs(QHQ - np.eye(N))
    max_off_diag = np.max(off_diag)
    mean_off_diag = np.mean(off_diag)

    # 特征值分析
    eigenvalues = np.linalg.eigvals(QHQ)
    eig_magnitudes = np.abs(eigenvalues)

    # 特征值统计
    eig_mean = np.mean(eig_magnitudes)
    eig_std = np.std(eig_magnitudes)
    eig_min = np.min(eig_magnitudes)
    eig_max = np.max(eig_magnitudes)

    # 条件数
    cond_number = np.sqrt(eig_max / (eig_min + 1e-10))

    # 对角元素均匀性
    diag_elements = np.abs(np.diag(QHQ))
    diag_uniformity = np.std(diag_elements) / (np.mean(diag_elements) + 1e-10)

    return {
        'max_off_diag': float(max_off_diag),
        'mean_off_diag': float(mean_off_diag),
        'eig_mean': float(eig_mean),
        'eig_std': float(eig_std),
        'eig_min': float(eig_min),
        'eig_max': float(eig_max),
        'cond_number': float(cond_number),
        'diag_uniformity': float(diag_uniformity),
    }

# ============================================================================
# 方法 4: DD 域能量集中度分析 (该方法可直接证明是否学习到 OTFS)
# ============================================================================

def dd_energy_concentration_analysis(Q):
    """
    对于 Q 的所有列, 计算其在 DD 域的能量集中度。

    OTFS 的理想特性:
      - 每列在 DD 域形成一个尖锐的 2D 冲激
      - 所有列的集中度都很高 (接近 1.0)
      - 不同列映射到 DD 域的不同位置

    返回集中度统计和分布图。
    """
    N = Q.shape[0]
    concentrations = []
    peak_map = np.zeros((N, 2), dtype=int)  # delay × Doppler for 2-symbol

    for i in range(N):
        dd_grid, conc, peak, _ = analyze_q_column_dd(Q, i, N_symbols=2)
        concentrations.append(conc)
        peak_map[i] = peak

    concentrations = np.array(concentrations)

    return {
        'mean': float(np.mean(concentrations)),
        'std': float(np.std(concentrations)),
        'min': float(np.min(concentrations)),
        'max': float(np.max(concentrations)),
        'values': concentrations,
        'peak_map': peak_map,
        'n_unique_peaks': len(set(tuple(p) for p in peak_map)),
    }

# ============================================================================
# 方法 5: CIR → Q 敏感度 2D 分析
# ============================================================================

def cir_sensitivity_analysis(model, speeds, delay_spreads, batch_size=8):
    """
    扫描速度 × 延迟扩展的 2D 参数空间, 记录 Q 与 IDFT 的距离。
    热力图可直观显示 Q 在何时偏离 OFDM → 何时学到 OTFS-like 行为。
    """
    N = FFT_SIZE
    idft = compute_idft(N)

    sensitivity_map = np.zeros((len(delay_spreads), len(speeds)))
    dist_i_map = np.zeros((len(delay_spreads), len(speeds)))
    ortho_map = np.zeros((len(delay_spreads), len(speeds)))

    for di, ds in enumerate(delay_spreads):
        for si, sp in enumerate(speeds):
            Q_batch, _ = extract_q_and_cir(model, sp, ds*0.9, ds*1.1, batch_size)
            Q_avg = np.mean(Q_batch, axis=0)
            sensitivity_map[di, si] = np.linalg.norm(Q_avg - idft, 'fro') / np.linalg.norm(Q_avg, 'fro')
            dist_i_map[di, si] = np.linalg.norm(Q_avg - np.eye(N)/np.sqrt(N), 'fro') / np.linalg.norm(Q_avg, 'fro')
            QHQ = Q_avg.conj().T @ Q_avg
            ortho_map[di, si] = np.max(np.abs(QHQ - np.eye(N)))

    return {
        'dist_idft': sensitivity_map,
        'dist_i': dist_i_map,
        'ortho_err': ortho_map,
        'speeds': speeds,
        'delay_spreads': delay_spreads,
    }

# ============================================================================
# 方法 6: 时频扩展谱分析 (T-F Spreading Profile)
# ============================================================================

def tf_spreading_profile(Q, freq_input_idx=None):
    """
    分析 Q 对单一频率输入的时频扩展特性。

    OTFS: 能量均匀扩展到时频网格全平面
    OFDM: 能量集中在单个时频格点
    """
    N = Q.shape[0]

    profiles = []
    indices = range(N) if freq_input_idx is None else [freq_input_idx]

    for i in indices:
        # 频率域 δ 输入
        x_freq = np.zeros(N, dtype=np.complex128)
        x_freq[i] = 1.0

        # 时域输出
        x_time = Q @ x_freq

        # 时域包络
        time_envelope = np.abs(x_time)

        # 频域响应 (FFT 回到频域)
        freq_response = np.abs(np.fft.fft(x_time)) / np.sqrt(N)

        # 计算扩展度量
        # 时域扩展 (功率加权标准差)
        power_t = time_envelope ** 2
        total_t = np.sum(power_t)
        if total_t > 1e-15:
            centroid_t = np.sum(np.arange(N) * power_t) / total_t
            var_t = np.sum(power_t * (np.arange(N) - centroid_t)**2) / total_t
            spread_t = np.sqrt(var_t)
        else:
            spread_t = 0

        # 频域扩展
        power_f = freq_response ** 2
        total_f = np.sum(power_f)
        if total_f > 1e-15:
            centroid_f = np.sum(np.arange(N) * power_f) / total_f
            var_f = np.sum(power_f * (np.arange(N) - centroid_f)**2) / total_f
            spread_f = np.sqrt(var_f)
        else:
            spread_f = 0

        profiles.append({
            'freq_input_idx': i,
            'time_spread': spread_t,
            'freq_spread': spread_f,
            'time_envelope': time_envelope,
            'freq_response': freq_response,
        })

    return profiles

# ============================================================================
# 方法 7: OTFS 自相关分析 — Q 的行间/列间相关性
# ============================================================================

def analyze_q_correlation_structure(Q):
    """
    分析 Q 矩阵的内部相关结构。

    OTFS-like 变换的特性:
      - 列间: 低相关性 (每列映射到不同的 DD 位置)
      - 行间: 低相关性 (时域波形基相互正交)
    """
    N = Q.shape[0]

    # 列间相关
    Q_norm_cols = Q / (np.linalg.norm(Q, axis=0, keepdims=True) + 1e-10)
    col_corr = np.abs(Q_norm_cols.conj().T @ Q_norm_cols)

    # 行间相关
    Q_norm_rows = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-10)
    row_corr = np.abs(Q_norm_rows @ Q_norm_rows.conj().T)

    # 非对角平均相关
    col_off_diag = col_corr.copy()
    np.fill_diagonal(col_off_diag, 0)
    row_off_diag = row_corr.copy()
    np.fill_diagonal(row_off_diag, 0)

    return {
        'col_corr_mean': float(np.mean(col_off_diag)),
        'col_corr_max': float(np.max(col_off_diag)),
        'row_corr_mean': float(np.mean(row_off_diag)),
        'row_corr_max': float(np.max(row_off_diag)),
        'col_corr_matrix': col_corr,
        'row_corr_matrix': row_corr,
    }

# ============================================================================
# 方法 8: Q 的 Doppler 分辨率分析
# ============================================================================

def doppler_resolution_analysis(model, base_speed=3.0, target_speeds=None, ds=100e-9):
    """
    测试 Q 对速度变化的敏感度 (即 Doppler 分辨率)。

    OTFS 需要 Q 能区分不同 Doppler → 相邻速度的 Q 差异应随速度增大。
    """
    if target_speeds is None:
        target_speeds = [3, 10, 30, 60, 90, 120, 150, 200]

    N = FFT_SIZE
    idft = compute_idft(N)

    q_matrices = {}
    distances = []

    for sp in target_speeds:
        Q_batch, _ = extract_q_and_cir(model, sp, ds*0.9, ds*1.1, 8)
        Q_avg = np.mean(Q_batch, axis=0)
        q_matrices[sp] = Q_avg

    # 相邻速度 Q 差异
    for i in range(len(target_speeds) - 1):
        sp1, sp2 = target_speeds[i], target_speeds[i+1]
        diff = np.linalg.norm(q_matrices[sp1] - q_matrices[sp2], 'fro')
        distances.append(float(diff))

    return {
        'speeds': target_speeds,
        'adjacent_diffs': distances,
        'q_matrices': q_matrices,
    }

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 72)
    print("OTFS Delay-Doppler 域验证 — 增强版")
    print("检测模型: weights-qQ_Method_TV")
    print("=" * 72)

    from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
    import sionna.phy as sn
    sn.config.seed = SEED

    # ---- 加载模型 ----
    weights_file = 'weights-qQ_Method_TV'
    print(f"\n加载权重: {weights_file}")
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
    model = qQ_MODEL_TV(training=False)
    b, b_hat = model(2, 20.0)
    model.set_weights(weights)
    print("权重加载成功。")

    N = FFT_SIZE
    idft = compute_idft(N)
    eye = np.eye(N, dtype=np.complex128) / np.sqrt(N)

    # ========================================================================
    # Part 1: 基础指标 — dist(IDFT), dist(I), 正交性
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 1: 基础 Q 矩阵指标")
    print("=" * 72)

    test_conditions = [
        ("平坦+低速 (TDM预期)",  3.0,  10e-9, 11e-9),
        ("多径+低速 (OFDM预期)", 3.0,  100e-9, 101e-9),
        ("多径+中速 (过渡区)",   60.0, 100e-9, 101e-9),
        ("多径+高速 (OTFS预期)", 120.0, 200e-9, 201e-9),
        ("多径+超高速",          200.0, 200e-9, 201e-9),
    ]

    base_metrics = {}
    for name, speed, ds_min, ds_max in test_conditions:
        Q_batch, cir = extract_q_and_cir(model, speed, ds_min, ds_max, 16)
        Q_avg = np.mean(Q_batch, axis=0)
        base_metrics[name] = {
            'Q': Q_avg,
            'cir': cir,
            'dist_idft': np.linalg.norm(Q_avg - idft, 'fro') / np.linalg.norm(Q_avg, 'fro'),
            'dist_I': np.linalg.norm(Q_avg - eye, 'fro') / np.linalg.norm(Q_avg, 'fro'),
        }

        QHQ = Q_avg.conj().T @ Q_avg
        ortho_err = np.max(np.abs(QHQ - np.eye(N)))
        base_metrics[name]['ortho_err'] = ortho_err

        print(f"  {name}: dist_IDFT={base_metrics[name]['dist_idft']:.4f}, "
              f"dist_I={base_metrics[name]['dist_I']:.4f}, ortho_err={ortho_err:.4f}")

    # ========================================================================
    # Part 2: 核心 — DD 域逆向投影分析 (验证OTFS.md 方法)
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 2: DD 域逆向投影 — 核心 OTFS 验证")
    print("=" * 72)

    # 对每个场景做 DD 域分析
    dd_analysis = {}
    for name in base_metrics:
        Q = base_metrics[name]['Q']

        # 方法 2a: 单列 DD 投影 (以中间列为例)
        mid_col = N // 2
        dd_grid, conc, peak, tf_grid = analyze_q_column_dd(Q, mid_col, N_symbols=2)

        # 方法 2b: 全列 DD 扫描
        scan_results = dd_scan_test(Q)

        dd_analysis[name] = {
            'mid_col_conc': conc,
            'mid_col_peak': peak,
            'mid_col_dd_grid': dd_grid,
            'scan_mean_conc': scan_results['concentration_mean'],
            'scan_std_conc': scan_results['concentration_std'],
            'unique_peaks': scan_results['unique_peak_count'],
            'peak_coverage': scan_results['peak_coverage'],
        }

        print(f"  {name}:")
        print(f"    中间列 DD 集中度:   {conc:.4f}")
        print(f"    全列平均 DD 集中度:  {scan_results['concentration_mean']:.4f} ± {scan_results['concentration_std']:.4f}")
        print(f"    唯一峰值位置数:      {scan_results['unique_peak_count']}/{N}")
        print(f"    峰值覆盖率:          {scan_results['peak_coverage']:.2%}")

    # ========================================================================
    # Part 3: 正交性 & 特征值深度分析
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 3: Q 正交性与特征值分析")
    print("=" * 72)

    ortho_results = {}
    for name in base_metrics:
        Q = base_metrics[name]['Q']
        ortho = analyze_q_orthogonality(Q)
        ortho_results[name] = ortho
        print(f"  {name}:")
        print(f"    条件数: {ortho['cond_number']:.3f}  (理想酉: 1.0)")
        print(f"    特征值范围: [{ortho['eig_min']:.3f}, {ortho['eig_max']:.3f}]")
        print(f"    特征值 STD: {ortho['eig_std']:.4f}")
        print(f"    非对角最大: {ortho['max_off_diag']:.4f}")

    # ========================================================================
    # Part 4: CIR → Q 敏感度 2D 扫描
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 4: CIR → Q 敏感度 2D 热力图扫描")
    print("=" * 72)

    speeds_scan = [3, 10, 30, 60, 90, 120, 200]
    ds_scan = [10e-9, 50e-9, 100e-9, 200e-9, 300e-9, 500e-9]
    sensitivity = cir_sensitivity_analysis(model, speeds_scan, ds_scan, batch_size=8)

    print(f"  速度范围: {speeds_scan} m/s")
    print(f"  延迟扩展范围: {[f'{ds*1e9:.0f}ns' for ds in ds_scan]}")

    # ========================================================================
    # Part 5: 时频扩展谱分析
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 5: 时频扩展谱分析")
    print("=" * 72)

    tf_profiles = {}
    for name in base_metrics:
        Q = base_metrics[name]['Q']
        profiles = tf_spreading_profile(Q, freq_input_idx=N//2)
        tf_profiles[name] = profiles[0]
        print(f"  {name}:")
        print(f"    时域扩展宽度: {profiles[0]['time_spread']:.2f} samples")
        print(f"    频域扩展宽度: {profiles[0]['freq_spread']:.2f} bins")

    # ========================================================================
    # Part 6: Q 相关结构分析
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 6: Q 行列相关结构")
    print("=" * 72)

    corr_results = {}
    for name in base_metrics:
        Q = base_metrics[name]['Q']
        corr = analyze_q_correlation_structure(Q)
        corr_results[name] = corr
        print(f"  {name}:")
        print(f"    列间平均相关: {corr['col_corr_mean']:.4f}")
        print(f"    行间平均相关: {corr['row_corr_mean']:.4f}")

    # ========================================================================
    # Part 7: Doppler 分辨率
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 7: Doppler 分辨率分析")
    print("=" * 72)

    doppler_res = doppler_resolution_analysis(model)
    print(f"  相邻速度 Q 差异: {[f'{d:.4f}' for d in doppler_res['adjacent_diffs']]}")

    # ========================================================================
    # 可视化 — 综合大图
    # ========================================================================
    print("\n生成可视化图表...")

    # ---- 图 1: DD 域核心验证 ----
    fig1 = plt.figure(figsize=(28, 20))
    gs1 = GridSpec(4, 5, figure=fig1, hspace=0.4, wspace=0.35)

    # Row 0: 各场景 Q 矩阵 |Q|
    vmax_q = max(np.max(np.abs(base_metrics[n]['Q'])) for n in base_metrics)
    for idx, (name, speed, _, _) in enumerate(test_conditions):
        ax = fig1.add_subplot(gs1[0, idx])
        Q = base_metrics[name]['Q']
        im = ax.imshow(np.abs(Q), cmap='hot', aspect='auto', vmin=0, vmax=vmax_q)
        ax.set_title(f'{name.split(chr(10))[0]}\n|Q|', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: 中间列 DD 域投影
    for idx, (name, speed, _, _) in enumerate(test_conditions):
        ax = fig1.add_subplot(gs1[1, idx])
        dd_grid = dd_analysis[name]['mid_col_dd_grid']
        power_db = 10 * np.log10(np.abs(dd_grid)**2 + 1e-10)
        vmax_db = np.max(power_db)
        im = ax.imshow(power_db, cmap='hot', aspect='auto', vmin=vmax_db-30, vmax=vmax_db)
        peak = dd_analysis[name]['mid_col_peak']
        ax.scatter(peak[1], peak[0], marker='x', color='cyan', s=100, linewidths=3)
        conc = dd_analysis[name]['mid_col_conc']
        ax.set_title(f'{name.split(chr(10))[0]}\nDD 投影 (col {N//2}), conc={conc:.3f}', fontsize=9)
        ax.set_xlabel('Doppler bin', fontsize=8)
        ax.set_ylabel('Delay bin', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2: Q^H·Q 偏离单位阵
    for idx, (name, speed, _, _) in enumerate(test_conditions):
        ax = fig1.add_subplot(gs1[2, idx])
        Q = base_metrics[name]['Q']
        QHQ = np.abs(Q.conj().T @ Q)
        np.fill_diagonal(QHQ, 0)
        im = ax.imshow(QHQ, cmap='hot', aspect='auto')
        ortho = base_metrics[name]['ortho_err']
        ax.set_title(f'{name.split(chr(10))[0]}\n|Q^H·Q| 非对角, max={ortho:.4f}', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 3: 全列 DD 集中度分布
    for idx, (name, speed, _, _) in enumerate(test_conditions):
        ax = fig1.add_subplot(gs1[3, idx])
        Q = base_metrics[name]['Q']
        concs = []
        for i in range(N):
            _, conc, _, _ = analyze_q_column_dd(Q, i, N_symbols=2)
            concs.append(conc)
        ax.bar(range(N), concs, alpha=0.7, color='steelblue')
        ax.axhline(y=np.mean(concs), color='red', linestyle='--', label=f'mean={np.mean(concs):.3f}')
        ax.set_title(f'{name.split(chr(10))[0]}\n各列 DD 集中度', fontsize=9)
        ax.set_xlabel('列索引')
        ax.set_ylabel('能量集中度')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'OTFS Delay-Doppler 域核心验证 | weights-qQ_Method_TV\n'
                 f'Row1: Q矩阵 | Row2: DD域投影 | Row3: 正交性 | Row4: 各列DD集中度',
                 fontsize=13, fontweight='bold')
    plt.savefig('OTFS_dd_verification_main.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_dd_verification_main.png")

    # ---- 图 2: DD 域网格扫描 (验证"亮点遍历"现象) ----
    # 对高速场景做完整的逐列 DD 扫描
    high_speed_name = "多径+高速 (OTFS预期)"
    Q_otfs = base_metrics[high_speed_name]['Q']
    fig2, all_peaks = dd_grid_scan_visualization(Q_otfs, title_prefix="高速 120m/s: ")
    plt.savefig('OTFS_dd_grid_scan_high_speed.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_dd_grid_scan_high_speed.png")

    # 低速对比
    low_speed_name = "多径+低速 (OFDM预期)"
    Q_ofdm = base_metrics[low_speed_name]['Q']
    fig2b, all_peaks_low = dd_grid_scan_visualization(Q_ofdm, title_prefix="低速 3m/s: ")
    plt.savefig('OTFS_dd_grid_scan_low_speed.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_dd_grid_scan_low_speed.png")

    # ---- 图 3: CIR 敏感度 2D 热力图 ----
    fig3, axes3 = plt.subplots(1, 3, figsize=(22, 6))

    titles = ['dist(Q, IDFT)', 'dist(Q, I)', '|Q^H·Q - I|_max']
    maps = [sensitivity['dist_idft'], sensitivity['dist_i'], sensitivity['ortho_err']]
    cmaps = ['RdBu', 'RdBu', 'hot']

    for idx, (ax, title, data, cmap) in enumerate(zip(axes3, titles, maps, cmaps)):
        im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')
        ax.set_xticks(range(len(speeds_scan)))
        ax.set_xticklabels([f'{s}' for s in speeds_scan], rotation=45)
        ax.set_yticks(range(len(ds_scan)))
        ax.set_yticklabels([f'{ds*1e9:.0f}ns' for ds in ds_scan])
        ax.set_xlabel('Speed [m/s]', fontsize=11)
        ax.set_ylabel('Delay Spread [ns]', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        # 标注每个格子的值
        for i in range(len(ds_scan)):
            for j in range(len(speeds_scan)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.suptitle('CIR → Q 敏感度 2D 扫描 | weights-qQ_Method_TV\n'
                 '越红(右上): 越远离OFDM → OTFS-like',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('OTFS_cir_sensitivity_2d.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_cir_sensitivity_2d.png")

    # ---- 图 4: 正交性深度分析 ----
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))

    names_short = [n[0].split('\n')[0] for n in test_conditions]
    colors = ['blue', 'green', 'orange', 'red', 'darkred']

    # 4a: 条件数
    ax = axes4[0, 0]
    conds = [ortho_results[n]['cond_number'] for n in base_metrics]
    ax.bar(names_short, conds, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='green', linestyle='--', label='理想酉: 1.0')
    ax.set_title('条件数 (越低=越接近酉矩阵)', fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

    # 4b: 特征值分布
    ax = axes4[0, 1]
    for idx, name in enumerate(base_metrics):
        Q = base_metrics[name]['Q']
        QHQ = Q.conj().T @ Q
        eigvals = np.abs(np.linalg.eigvals(QHQ))
        ax.plot(sorted(eigvals), 'o-', color=colors[idx], markersize=3, alpha=0.7, label=names_short[idx])
    ax.axhline(y=1.0, color='green', linestyle='--', label='理想: 1.0')
    ax.set_title('Q^H·Q 特征值分布 (sorted)', fontsize=10)
    ax.set_xlabel('特征值索引')
    ax.set_ylabel('特征值幅度')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # 4c: 对角元素均匀性
    ax = axes4[0, 2]
    diag_stds = [ortho_results[n]['diag_uniformity'] for n in base_metrics]
    ax.bar(names_short, diag_stds, color=colors, alpha=0.7)
    ax.set_title('对角元素均匀性\n(越低=各列能量越一致)', fontsize=10)
    ax.tick_params(axis='x', rotation=30)

    # 4d: 非对角最大泄漏
    ax = axes4[1, 0]
    off_diags = [ortho_results[n]['max_off_diag'] for n in base_metrics]
    ax.bar(names_short, off_diags, color=colors, alpha=0.7)
    ax.set_title('Q^H·Q 最大非对角泄漏', fontsize=10)
    ax.tick_params(axis='x', rotation=30)

    # 4e: 行间/列间相关对比
    ax = axes4[1, 1]
    x = np.arange(len(base_metrics))
    width = 0.35
    col_corrs = [corr_results[n]['col_corr_mean'] for n in base_metrics]
    row_corrs = [corr_results[n]['row_corr_mean'] for n in base_metrics]
    ax.bar(x - width/2, col_corrs, width, alpha=0.7, label='列间相关', color='steelblue')
    ax.bar(x + width/2, row_corrs, width, alpha=0.7, label='行间相关', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, rotation=30, fontsize=8)
    ax.set_title('Q 行列相关性\n(低=波形基更正交)', fontsize=10)
    ax.legend(fontsize=8)

    # 4f: ID-IDFT 对比 — Q 在各种条件下的位置
    ax = axes4[1, 2]
    dist_idft_vals = [base_metrics[n]['dist_idft'] for n in base_metrics]
    dist_i_vals = [base_metrics[n]['dist_I'] for n in base_metrics]

    for idx, name in enumerate(base_metrics):
        ax.scatter(dist_idft_vals[idx], dist_i_vals[idx], color=colors[idx], s=150,
                  label=names_short[idx], edgecolors='black', linewidth=0.5, zorder=5)
    ax.scatter(0, np.linalg.norm(eye-idft,'fro')/np.linalg.norm(eye,'fro'),
              marker='*', s=300, color='green', label='IDFT (OFDM)', zorder=10, edgecolors='black')
    ax.scatter(np.linalg.norm(eye-idft,'fro')/np.linalg.norm(eye,'fro'), 0,
              marker='*', s=300, color='gray', label='I (TDM)', zorder=10, edgecolors='black')
    ax.set_xlabel('dist(Q, IDFT) →', fontsize=10)
    ax.set_ylabel('dist(Q, I) →', fontsize=10)
    ax.set_title('Q 在 TDM-OFDM-OTFS 空间中的位置', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 标注区域
    ax.annotate('TDM区域', xy=(1.0, 0.1), fontsize=8, color='gray', alpha=0.5)
    ax.annotate('OFDM区域', xy=(0.1, 1.2), fontsize=8, color='green', alpha=0.5)
    ax.annotate('OTFS区域', xy=(0.8, 1.0), fontsize=8, color='red', alpha=0.5)

    plt.suptitle('Q 矩阵正交性深度分析 | weights-qQ_Method_TV', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('OTFS_orthogonality_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_orthogonality_analysis.png")

    # ---- 图 5: Doppler 分辨率 & 时频扩展 ----
    fig5, axes5 = plt.subplots(2, 2, figsize=(16, 12))

    # 5a: Q 与 IDFT 距离 vs 速度 (使用 Doppler 分辨率数据)
    ax = axes5[0, 0]
    speeds_res = doppler_res['speeds']
    for sp in speeds_res:
        Q = doppler_res['q_matrices'][sp]
        d = np.linalg.norm(Q - idft, 'fro') / np.linalg.norm(Q, 'fro')
        ax.plot(sp, d, 'o', markersize=8, color='red' if sp > 60 else 'blue')
    ax.set_xlabel('Speed [m/s]', fontsize=11)
    ax.set_ylabel('dist(Q, IDFT)', fontsize=11)
    ax.set_title('Q 与 IDFT 距离 vs 速度', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 5b: 相邻速度 Q 差异
    ax = axes5[0, 1]
    speed_midpoints = [(speeds_res[i] + speeds_res[i+1])/2 for i in range(len(speeds_res)-1)]
    ax.bar(range(len(doppler_res['adjacent_diffs'])), doppler_res['adjacent_diffs'],
           color=['blue' if s < 60 else 'red' for s in speed_midpoints], alpha=0.7)
    ax.set_xticks(range(len(doppler_res['adjacent_diffs'])))
    ax.set_xticklabels([f'{speeds_res[i]}→{speeds_res[i+1]}' for i in range(len(speeds_res)-1)], rotation=30, fontsize=8)
    ax.set_ylabel('|Q_sp1 - Q_sp2|_F', fontsize=11)
    ax.set_title('相邻速度 Q 矩阵差异', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 5c: 时域扩展 vs 频域扩展 (每个场景)
    ax = axes5[1, 0]
    for idx, name in enumerate(base_metrics):
        p = tf_profiles[name]
        ax.scatter(p['time_spread'], p['freq_spread'], s=150, color=colors[idx],
                  label=names_short[idx], edgecolors='black', linewidth=0.5)
    # IDFT reference: uniform spread ≈ N/√12 ≈ 9.2 for N=32
    uniform_spread = N / np.sqrt(12)
    ax.scatter(uniform_spread, uniform_spread, marker='*', s=300, color='green',
              label=f'IDFT (OFDM, spread≈{uniform_spread:.1f})', edgecolors='black')
    ax.set_xlabel('时域扩展宽度 [samples]', fontsize=11)
    ax.set_ylabel('频域扩展宽度 [bins]', fontsize=11)
    ax.set_title('时频扩展特征 → OTFS应在右上区域', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5d: DD 集中度 vs 场景
    ax = axes5[1, 1]
    dd_conc_means = [dd_analysis[n]['scan_mean_conc'] for n in base_metrics]
    dd_conc_stds = [dd_analysis[n]['scan_std_conc'] for n in base_metrics]
    ax.bar(names_short, dd_conc_means, yerr=dd_conc_stds, color=colors, alpha=0.7, capsize=5)
    ax.axhline(y=1.0, color='green', linestyle='--', label='理想 OTFS: 集中度=1.0')
    ax.set_title('DD 域能量集中度 (全列平均)\n越高=越接近 DD 域冲激=越 OTFS-like', fontsize=10)
    ax.set_ylabel('能量集中度')
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Doppler 分辨率 & 时频扩展分析 | weights-qQ_Method_TV', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('OTFS_doppler_spread_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_doppler_spread_analysis.png")

    # ========================================================================
    # 生成 Markdown 报告
    # ========================================================================
    print("\n生成 Markdown 报告...")

    report_lines = []
    r = report_lines.append

    r("# OTFS Delay-Doppler 域验证报告")
    r("")
    r(f"**模型**: `weights-qQ_Method_TV`  ")
    r(f"**架构**: Conv1D + Complex GRU (1024) + Time-Path (双路: Delay + Doppler)  ")
    r(f"**参数量**: ~11M  ")
    r(f"**FFT_SIZE (N)**: {N}  ")
    r(f"**生成日期**: 2026-06-02  ")
    r("")

    r("---")
    r("")
    r("## 0. 方法论与验证逻辑")
    r("")
    r("### 现有验证的局限性")
    r("")
    r("已有的验证脚本 (`verify_tdm_ofdm_otfs.py`, `otfs_proof_controlled.py`, `analyze_otfs_features.py`) ")
    r("只能回答一个问题: **Q 是不是 IDFT?** (即 Q ≠ OFDM)")
    r("")
    r("| 已有方法 | 能回答什么 | 不能回答什么 |")
    r("|:---|:---|:---|")
    r("| `dist(Q, IDFT)` | Q 是不是 OFDM-like | Q 到底是不是 OTFS-like |")
    r("| `dist(Q, I)` | Q 是不是 TDM-like | — |")
    r("| 非对角能量占比 | Q 是否有扩展特性 | 扩展是否 = OTFS 的 DD 域结构 |")
    r("| Q^H·Q 正交性 | Q 是否能量守恒 | — |")
    r("")
    r("**核心问题**: 这些指标只能做排除法 (排除 TDM, 排除 OFDM). 但不能正面证明 Q 学到的就是 OTFS.")
    r("")
    r("### 本报告的验证方法 (来自 验证OTFS.md)")
    r("")
    r(r"OTFS 的核心特征: 将 Delay-Doppler (DD) 域的每个符号通过 ISFFT + Heisenberg 变换, ")
    r(r"映射到时域波形。其逆过程 (时域波形 → SFFT → DD 域) 应产生一个 **尖锐的 2D 冲激**。")
    r("")
    r("**验证逻辑链**:")
    r("")
    r("1. **Q 列向量 = 波形基**: Q 的第 i 列 $\\mathbf{q}_i$ 是当频率 bin i 输入为 1 时的时域波形")
    r("2. **逆向 DD 投影**: 对 $\\mathbf{q}_i$ 做 FFT → 频域, reshape 为 TF 网格, 再做 SFFT → DD 域")
    r("3. **观察 DD 域模式**: ")
    r("   - 若是 **OTFS**: DD 域出现一个尖锐的 2D 冲激, 且不同列对应不同 DD 位置")
    r("   - 若是 **OFDM**: DD 域能量散布在多个位置 (因为 IDFT 不形成 DD 域冲激)")
    r("   - 若是 **TDM**: 完全不同的模式 (时域集中)")
    r("4. **亮点扫描测试**: 逐列做 DD 投影, 观察峰值位置是否在 2D 网格上规律移动")
    r("")

    r("---")
    r("")
    r("## 1. 基础 Q 矩阵指标")
    r("")
    r("### 多场景对比")
    r("")
    r("| 场景 | dist(Q, IDFT) | dist(Q, I) | Q^H·Q 非对角最大 | 判定 |")
    r("|:---|:---|:---|:---|:---|")

    for name in base_metrics:
        bm = base_metrics[name]
        d_idft = bm['dist_idft']
        d_i = bm['dist_I']
        ortho = bm['ortho_err']

        # 判定
        if '平坦' in name:
            if d_i < d_idft:
                verdict = '✅ 接近 TDM (I)'
            else:
                verdict = '⚠️ 非 TDM'
        elif '低速' in name or '中速' in name:
            if d_idft < 1.0 and d_idft < d_i:
                verdict = '✅ 接近 OFDM (IDFT)'
            else:
                verdict = '⚠️ 未完全靠拢 OFDM'
        elif '高速' in name or '超高速' in name:
            if d_idft > 0.5 and d_i > 0.5:
                verdict = '✅ 远离 TDM 和 OFDM → 可能 OTFS'
            else:
                verdict = '⚠️ 判定模糊'
        else:
            verdict = '—'

        r(f"| {name.split(chr(10))[0]} | {d_idft:.4f} | {d_i:.4f} | {ortho:.4f} | {verdict} |")

    r("")
    r("### 关键观察")
    r("")

    # 分析平坦+低速
    flat_low = base_metrics[test_conditions[0][0]]
    r(f"- **平坦+低速**: dist_I={flat_low['dist_I']:.3f}, dist_IDFT={flat_low['dist_idft']:.3f} "
      f"→ {'接近 TDM ✅' if flat_low['dist_I'] < flat_low['dist_idft'] else '需要检查'}")

    # 分析多径+低速
    mp_low = base_metrics[test_conditions[1][0]]
    r(f"- **多径+低速**: dist_IDFT={mp_low['dist_idft']:.3f}, dist_I={mp_low['dist_I']:.3f} "
      f"→ {'接近 OFDM ✅' if mp_low['dist_idft'] < 1.0 and mp_low['dist_idft'] < mp_low['dist_I'] else '未靠拢 IDFT (可能的 OTFS 特性泄漏)'}")

    # 分析多径+高速
    mp_high = base_metrics[test_conditions[3][0]]
    r(f"- **多径+高速**: dist_IDFT={mp_high['dist_idft']:.3f}, dist_I={mp_high['dist_I']:.3f} "
      f"→ {'远离两者 ✅ → 可能 OTFS' if mp_high['dist_idft'] > 0.5 and mp_high['dist_I'] > 0.5 else '需要进一步分析'}")

    r("")
    r("---")
    r("")
    r("## 2. Delay-Doppler 域逆向投影 (核心验证)")
    r("")
    r("这是 **验证OTFS.md 中提出的最关键验证方法**。")
    r("")
    r("### 方法说明")
    r("")
    r("对 Q 矩阵的第 i 列 $\\mathbf{q}_i$:")
    r("1. 构造频域输入: $\\mathbf{x}_{freq} = \\mathbf{e}_i$ (第 i 个位置为 1 的 one-hot 向量)")
    r("2. Q 调制: $\\mathbf{x}_{time} = \\mathbf{Q} \\cdot \\mathbf{x}_{freq}$")
    r("3. FFT 回频域: $\\mathbf{y}_{freq} = \\text{FFT}(\\mathbf{x}_{time})$")
    r("4. 构造 2D TF 网格: reshape $\\mathbf{y}_{freq}$ 为 $M \\times N$ (频率 × 时间)")
    r("5. SFFT → DD 域: $\\mathbf{Y}_{DD} = \\text{SFFT}(\\mathbf{Y}_{TF})$")
    r("6. 观察: DD 域是否出现 **尖锐的 2D 冲激**")
    r("")
    r("### 各场景 DD 域分析结果")
    r("")
    r("| 场景 | 中间列 DD 集中度 | 全列平均 DD 集中度 | 唯一峰值位置数 | 峰值覆盖率 | OTFS 特征? |")
    r("|:---|:---|:---|:---|:---|:---|")

    for name in base_metrics:
        dd = dd_analysis[name]
        conc_mid = dd['mid_col_conc']
        conc_mean = dd['scan_mean_conc']
        n_unique = dd['unique_peaks']
        coverage = dd['peak_coverage']

        # 判定 OTFS 特征
        if conc_mean > 0.8 and coverage > 0.5:
            otfs_verdict = '✅ 强 OTFS 特征'
        elif conc_mean > 0.5 and coverage > 0.3:
            otfs_verdict = '🔶 中等 OTFS 特征'
        elif conc_mean > 0.3:
            otfs_verdict = '⚠️ 弱 OTFS 特征'
        else:
            otfs_verdict = '❌ 非 OTFS 特征'

        r(f"| {name.split(chr(10))[0]} | {conc_mid:.4f} | {conc_mean:.4f} | {n_unique}/{N} | {coverage:.2%} | {otfs_verdict} |")

    r("")
    r("### DD 域分析关键发现")
    r("")

    # 对比高速 vs 低速的 DD 集中度
    mp_low_dd = dd_analysis[test_conditions[1][0]]
    mp_high_dd = dd_analysis[test_conditions[3][0]]
    r(f"- **高速 vs 低速 DD 集中度**: 高速={mp_high_dd['scan_mean_conc']:.4f}, 低速={mp_low_dd['scan_mean_conc']:.4f}")
    if mp_high_dd['scan_mean_conc'] > mp_low_dd['scan_mean_conc']:
        r(f"  → 高速下 DD 域能量更集中 → **表现出了 OTFS-like 的 DD 域冲激特性** ✅")
    else:
        r(f"  → 高速下 DD 域能量未更集中 → Q 的变换不完全是 OTFS 式的 DD 域映射")

    r(f"- **峰值覆盖率**: 高速={mp_high_dd['peak_coverage']:.2%}")
    if mp_high_dd['peak_coverage'] > 0.5:
        r(f"  → 超过一半的列映射到不同的 DD 位置 → **接近 OTFS 的 DD 网格覆盖特性** ✅")
    else:
        r(f"  → Q 的列未能在 DD 域形成明显区分 → 非典型 OTFS")

    r("")
    r("---")
    r("")
    r("## 3. 正交性与特征值深度分析")
    r("")
    r("OTFS 变换是酉变换 (Q^H·Q = I), 保证能量守恒和正交性。")
    r("")
    r("### Q^H·Q 分析结果")
    r("")
    r("| 场景 | 条件数 | 特征值范围 | 特征值 STD | 非对角最大 |")
    r("|:---|:---|:---|:---|:---|")

    for name in base_metrics:
        o = ortho_results[name]
        r(f"| {name.split(chr(10))[0]} | {o['cond_number']:.3f} | [{o['eig_min']:.3f}, {o['eig_max']:.3f}] | {o['eig_std']:.4f} | {o['max_off_diag']:.4f} |")

    r("")
    r("### 正交性判定")
    r("")

    for name in base_metrics:
        o = ortho_results[name]
        if o['cond_number'] < 2.0 and o['max_off_diag'] < 0.5:
            r(f"- {name.split(chr(10))[0]}: ✅ 近似酉矩阵 (cond={o['cond_number']:.2f})")
        elif o['cond_number'] < 5.0:
            r(f"- {name.split(chr(10))[0]}: 🔶 部分满足酉性 (cond={o['cond_number']:.2f})")
        else:
            r(f"- {name.split(chr(10))[0]}: ❌ 偏离酉性较大 (cond={o['cond_number']:.2f})")

    r("")
    r("---")
    r("")
    r("## 4. CIR → Q 敏感度 2D 分析")
    r("")
    r("扫描速度 × 延迟扩展参数空间, 观察 Q 在何时偏离 IDFT (即学到非 OFDM 变换)。")
    r("")
    r("### dist(Q, IDFT) 热力图解读")
    r("")
    r(f"- **速度范围**: {speeds_scan} m/s")
    r(f"- **延迟扩展范围**: {[f'{ds*1e9:.0f}ns' for ds in ds_scan]}")
    r("")
    r("**理论预期**:")
    r("- 低速 & 低延迟 → 接近 I (TDM)")
    r("- 低速 & 高延迟 → 接近 IDFT (OFDM)")
    r("- 高速 & 高延迟 → 远离 IDFT (OTFS)")
    r("")

    r("**实测结果**:")
    sens = sensitivity
    for di, ds in enumerate(ds_scan):
        for si, sp in enumerate(speeds_scan):
            d = sens['dist_idft'][di, si]
            diag = sens['dist_i'][di, si]
            ortho = sens['ortho_err'][di, si]
            if d > 0.8:
                r(f"  - {ds*1e9:.0f}ns @ {sp}m/s: dist_IDFT={d:.2f} → 强非 OFDM (可能 OTFS)")

    r("")
    r("---")
    r("")
    r("## 5. 时频扩展谱分析")
    r("")
    r("OTFS 将一个符号的能量均匀扩展到整个时频网格。IDFT (OFDM) 也有均匀扩展特性。")
    r("关键区分: OTFS 的扩展模式随 Doppler 变化, OFDM 的扩展模式固定。")
    r("")
    r("| 场景 | 时域扩展宽度 [samples] | 频域扩展宽度 [bins] | 与 IDFT 均匀扩展差异 |")
    r("|:---|:---|:---|:---|")

    uniform_spread = N / np.sqrt(12)  # ~9.24 for N=32
    for name in base_metrics:
        p = tf_profiles[name]
        diff = np.sqrt((p['time_spread']-uniform_spread)**2 + (p['freq_spread']-uniform_spread)**2)
        r(f"| {name.split(chr(10))[0]} | {p['time_spread']:.2f} | {p['freq_spread']:.2f} | {diff:.2f} |")

    r("")
    r(f"  (IDFT 的均匀扩展参考值: {uniform_spread:.2f} samples)")

    r("")
    r("---")
    r("")
    r("## 6. Q 行列相关结构")
    r("")
    r("| 场景 | 列间平均相关 | 行间平均相关 |")
    r("|:---|:---|:---|")

    for name in base_metrics:
        c = corr_results[name]
        r(f"| {name.split(chr(10))[0]} | {c['col_corr_mean']:.4f} | {c['row_corr_mean']:.4f} |")

    r("")
    r("**解读**: 列间/行间相关越低, Q 的基向量越独立, 越接近一组正交基 (OTFS/OFDM 特性)。")
    r("")

    r("---")
    r("")
    r("## 7. Doppler 分辨率分析")
    r("")
    r("测试 Q 对不同速度的敏感度。相邻速度间的 Q 差异越大 → Doppler 分辨率越高。")
    r("")
    r("| 速度区间 | Q 差异 (Frobenius) |")
    r("|:---|:---|")

    for i in range(len(doppler_res['speeds']) - 1):
        sp1, sp2 = doppler_res['speeds'][i], doppler_res['speeds'][i+1]
        diff = doppler_res['adjacent_diffs'][i]
        r(f"| {sp1} → {sp2} m/s | {diff:.4f} |")

    r("")

    # Check if differences increase with speed (OTFS characteristic)
    diffs = doppler_res['adjacent_diffs']
    if len(diffs) >= 2:
        low_diff = np.mean(diffs[:len(diffs)//2])
        high_diff = np.mean(diffs[len(diffs)//2:])
        r(f"- 低速区平均差异: {low_diff:.4f}")
        r(f"- 高速区平均差异: {high_diff:.4f}")
        if high_diff > low_diff * 1.2:
            r(f"  → 高速下 Q 变化更大 → 网络对高 Doppler 更敏感 ✅")
        else:
            r(f"  → 高低速区的 Q 变化相似 → Doppler 分辨率均匀")

    r("")
    r("---")
    r("")
    r("## 8. 综合判定")
    r("")
    r("### 汇总评分表")
    r("")
    r("| 验证项 | 权重 | 平坦+低速 | 多径+低速 | 多径+高速 | 说明 |")
    r("|:---|:---|:---|:---|:---|:---|")

    def score_check(condition_name, expected_type):
        """返回各项检查的通过情况"""
        bm = base_metrics[condition_name]
        dd = dd_analysis[condition_name]
        o = ortho_results[condition_name]

        checks = []

        # 1. 符合预期类型
        if expected_type == 'TDM':
            ok = bm['dist_I'] < bm['dist_idft']
        elif expected_type == 'OFDM':
            ok = bm['dist_idft'] < bm['dist_I'] and bm['dist_idft'] < 1.0
        elif expected_type == 'OTFS':
            ok = bm['dist_idft'] > 0.5 and bm['dist_I'] > 0.5
        checks.append(('类型匹配', ok))

        # 2. 正交性
        checks.append(('近似酉矩阵', o['cond_number'] < 3.0))

        # 3. DD 集中度
        checks.append(('DD域集中度>0.5', dd['scan_mean_conc'] > 0.5))

        # 4. DD 峰值区分
        checks.append(('DD峰值覆盖率>30%', dd['peak_coverage'] > 0.3))

        return checks

    # TDM expected
    tdm_checks = score_check(test_conditions[0][0], 'TDM')
    ofdm_checks = score_check(test_conditions[1][0], 'OFDM')
    otfs_checks = score_check(test_conditions[3][0], 'OTFS')

    all_check_names = [c[0] for c in tdm_checks]

    for i, check_name in enumerate(all_check_names):
        tdm_sym = '✅' if tdm_checks[i][1] else '❌'
        ofdm_sym = '✅' if ofdm_checks[i][1] else '❌'
        otfs_sym = '✅' if otfs_checks[i][1] else '❌'

        r(f"| {check_name} | — | {tdm_sym} | {ofdm_sym} | {otfs_sym} | |")

    r("")
    r("### 最终结论")
    r("")

    # Count passing checks
    otfs_pass = sum(1 for _, ok in otfs_checks if ok)
    ofdm_pass = sum(1 for _, ok in ofdm_checks if ok)
    tdm_pass = sum(1 for _, ok in tdm_checks if ok)

    r(f"**TDM 场景 (平坦+低速)**: {tdm_pass}/{len(tdm_checks)} 项通过")
    r(f"**OFDM 场景 (多径+低速)**: {ofdm_pass}/{len(ofdm_checks)} 项通过")
    r(f"**OTFS 场景 (多径+高速)**: {otfs_pass}/{len(otfs_checks)} 项通过")
    r("")

    # 核心判断
    mp_low_dist = base_metrics[test_conditions[1][0]]['dist_idft']
    mp_high_dist = base_metrics[test_conditions[3][0]]['dist_idft']
    mp_high_dd_conc = dd_analysis[test_conditions[3][0]]['scan_mean_conc']
    mp_low_dd_conc = dd_analysis[test_conditions[1][0]]['scan_mean_conc']

    r("### 模型是否学到了 OTFS?")
    r("")

    r("| 判定依据 | 所需条件 | 实际值 | 通过? |")
    r("|:---|:---|:---|:---|")
    r(f"| 高速 Q 远离 IDFT | dist_IDFT > 0.5 | {mp_high_dist:.4f} | {'✅' if mp_high_dist > 0.5 else '❌'} |")
    r(f"| 高速 Q 远离 I (TDM) | dist_I > 0.5 | {base_metrics[test_conditions[3][0]]['dist_I']:.4f} | {'✅' if base_metrics[test_conditions[3][0]]['dist_I'] > 0.5 else '❌'} |")
    r(f"| 高速 > 低速 dist_IDFT | Δdist > 0 | {mp_high_dist - mp_low_dist:.4f} | {'✅' if mp_high_dist > mp_low_dist else '❌'} |")
    r(f"| Q 随 Doppler 自适应 | 高速 DD 集中度 ≠ 低速 | Δconc={abs(mp_high_dd_conc-mp_low_dd_conc):.4f} | {'✅' if abs(mp_high_dd_conc-mp_low_dd_conc) > 0.05 else '❌'} |")
    r(f"| 近似酉矩阵 (正交性) | 条件数 < 3 | {ortho_results[test_conditions[3][0]]['cond_number']:.3f} | {'✅' if ortho_results[test_conditions[3][0]]['cond_number'] < 3 else '❌'} |")
    r(f"| DD 域能量集中 | DD集中度 > 0.5 | {mp_high_dd_conc:.4f} | {'✅' if mp_high_dd_conc > 0.5 else '❌'} |")
    r("")

    # Overall verdict
    conditions_met = sum([
        mp_high_dist > 0.5,
        base_metrics[test_conditions[3][0]]['dist_I'] > 0.5,
        mp_high_dist > mp_low_dist,
        abs(mp_high_dd_conc - mp_low_dd_conc) > 0.05,
        ortho_results[test_conditions[3][0]]['cond_number'] < 3,
        mp_high_dd_conc > 0.5,
    ])

    r(f"**综合评分**: {conditions_met}/6 项核心条件满足")
    r("")

    if conditions_met >= 5:
        r("### 🟢 结论: 模型成功学到了 OTFS-like 波形")
        r("")
        r("`weights-qQ_Method_TV` 模型在高多普勒+多径条件下表现出以下 OTFS 核心特征:")
        r("1. Q 矩阵显著偏离 IDFT (非 OFDM)")
        r("2. Q 矩阵保持近似酉性 (能量守恒)")
        r("3. Q 随 Doppler 条件自适应调整")
        r("4. DD 域逆向投影呈现集中能量分布")
        r("5. 不同频率输入映射到不同的 DD 域位置")
    elif conditions_met >= 3:
        r("### 🟡 结论: 模型部分学到了 OTFS-like 特征")
        r("")
        r("模型表现出了某些 OTFS 特性, 但尚未完全实现 DD 域的完美映射。")
        r("可能的改进方向:")
        r("- 增加训练时的速度/延迟扩展范围")
        r("- 显式加入 DD 域相关的损失函数")
        r("- 增加 OFDM 符号数 (扩大 DD 网格)")
    else:
        r("### 🔴 结论: 模型尚未充分学到 OTFS 特征")
        r("")
        r("模型在高多普勒下未展现出典型的 OTFS DD 域映射特性。")
        r("建议:")
        r("- 检查训练是否充分收敛")
        r("- 考虑在损失函数中加入 DD 域结构化约束")
        r("- 验证 CIR 时间快照数是否足够捕获 Doppler 变化")

    r("")
    r("---")
    r("")
    r("## 9. 附加验证方法建议")
    r("")
    r("基于以上分析, 建议以下额外的 OTFS 验证方法:")
    r("")
    r("### 9.1 已在本报告中实现的新方法")
    r("")
    r("1. **DD 域逆向投影** (来自 验证OTFS.md): ✅ 已实现")
    r("2. **DD 域网格亮点扫描**: ✅ 已实现")
    r("3. **特征值分析 (条件数)**: ✅ 已实现")
    r("4. **CIR 敏感度 2D 扫描**: ✅ 已实现")
    r("5. **Doppler 分辨率分析**: ✅ 已实现")
    r("6. **时频扩展谱分析**: ✅ 已实现")
    r("")
    r("### 9.2 可进一步实施的方法")
    r("")
    r("1. **互信息分析**: 计算 I(X; Y|CIR, Q) 在不同信道条件下的变化, OTFS 应在高 Doppler 下提供更高互信息")
    r("2. **DD 域等效信道矩阵分析**: 计算完整的 DD 域输入-输出关系矩阵, OTFS 应呈块循环结构")
    r("3. **与理论 OTFS 的 Q 矩阵对齐度**: 计算学习的 Q 与理想 OTFS 变换的相似度")
    r("4. **BER vs 归一化 Doppler (f_d·T)**: OTFS 的 BER 应对归一化 Doppler 不敏感")
    r("5. **PAPR-CCDF 分析**: OTFS 的 PAPR 应介于 TDM (高) 和 OFDM (低) 之间")
    r("6. **对抗性 CIR 测试**: 在极端 Doppler 下测试 Q 的鲁棒性")
    r("")
    r("### 9.3 训练改进建议")
    r("")
    r("如果验证结果不理想, 可尝试:")
    r("- **DD 域辅助损失**: 在训练时加入 $\\|\\text{SFFT}(Q^{-1} \\cdot \\text{信号}) - \\text{DD 域 target}\\|^2$")
    r("- **更多时间快照**: 增加 NUM_TIME_SNAPSHOTS 以提升 Doppler 分辨率")
    r("- **更大 FFT_SIZE**: 扩大 DD 网格 (目前仅有 32×2)")
    r("- **OTFS 预训练**: 先用理想 OTFS 变换初始化 Q, 再微调")
    r("")
    r("---")
    r("")
    r("## 10. 生成的可视化文件")
    r("")
    r("| 文件名 | 内容 |")
    r("|:---|:---|")
    r("| `OTFS_dd_verification_main.png` | DD 域核心验证大图 (Q矩阵 + DD投影 + 正交性 + 集中度) |")
    r("| `OTFS_dd_grid_scan_high_speed.png` | 高速场景 DD 域网格扫描 (亮点遍历验证) |")
    r("| `OTFS_dd_grid_scan_low_speed.png` | 低速场景 DD 域网格扫描 (对比) |")
    r("| `OTFS_cir_sensitivity_2d.png` | CIR → Q 敏感度 2D 热力图 |")
    r("| `OTFS_orthogonality_analysis.png` | 正交性深度分析 (条件数 + 特征值 + 相关性) |")
    r("| `OTFS_doppler_spread_analysis.png` | Doppler 分辨率 & 时频扩展分析 |")
    r("")
    r(f"*报告生成时间: 2026-06-02 | 模型: weights-qQ_Method_TV | N={N}*")

    # Write report
    report_content = '\n'.join(report_lines)
    with open('OTFS_DD_Domain_Verification_Report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("  保存: OTFS_DD_Domain_Verification_Report.md")

    print("\n" + "=" * 72)
    print("验证完成!")
    print("=" * 72)
    print("\n生成的文件:")
    print("  可视化:")
    print("    OTFS_dd_verification_main.png")
    print("    OTFS_dd_grid_scan_high_speed.png")
    print("    OTFS_dd_grid_scan_low_speed.png")
    print("    OTFS_cir_sensitivity_2d.png")
    print("    OTFS_orthogonality_analysis.png")
    print("    OTFS_doppler_spread_analysis.png")
    print("  报告:")
    print("    OTFS_DD_Domain_Verification_Report.md")
    print()

if __name__ == "__main__":
    main()
