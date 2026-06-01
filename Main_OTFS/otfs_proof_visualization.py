#!/usr/bin/env python3
"""
OTFS 特征直观证明脚本
======================
OFDM 证明思路：看基底是不是不同频率的正弦波 → 看频谱是不是多载波
OTFS 证明思路（对应）：
  1. Q 矩阵是不是"密集扩展矩阵"（非对角 = 非 OFDM，每个符号跨多个时间采样）
  2. 单符号时频响应：发送一个孤立符号，看能量是否扩展到时频网格全平面
  3. Q 的西性 (Unitarity)：Q·Q^H ≈ I（保证能量守恒，如 OTFS 的正交性）
  4. Q 与 IDFT 矩阵的比较：距离 IDFT 越远 = 越不是 OFDM
  5. 高低 Doppler 下 Q 的结构变化（自适应扩展模式）
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
# ---- 中文字体配置 ----
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import *

def build_channel(speed_min, speed_max):
    from utils.TDL_RandomDS import TDL_RandomDS
    return TDL_RandomDS(
        model='A', delay_spread_min=50e-9, delay_spread_max=300e-9,
        carrier_frequency=CARRIER_FREQ, min_speed=speed_min, max_speed=speed_max
    )

def extract_q_matrix(model, speed, ebno_db=20.0):
    """Extract Q matrices at specific speed."""
    from sionna.phy.channel import cir_to_time_channel
    original_channel = model._channel_model
    test_channel = build_channel(max(0.1, speed - 0.5), speed + 0.5)
    model._channel_model = test_channel

    batch_size = 8
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
    pilots = tf.transpose(pilots, [0, 2, 1])

    creator = model._qQ_creator_layer
    Q, _ = creator(pilots, training=False)

    model._channel_model = original_channel
    return Q.numpy()

def compute_idft_matrix(N):
    """Standard IDFT matrix: F[k,n] = exp(j*2π*k*n/N) / sqrt(N)"""
    k = np.arange(N)
    n = np.arange(N).reshape(-1, 1)
    F = np.exp(1j * 2 * np.pi * k * n / N) / np.sqrt(N)
    return F

def compute_isotfft_matrix(N):
    """
    Ideal OTFS ISFFT-like transform.
    Symplectic Fourier: maps delay-Doppler → time-frequency.
    Returns the 2D transformation that spreads each DD symbol.
    For N×1 grid (simplified to 1D delay spread):
    Z[k,l] = 1/sqrt(N) * sum_n sum_m X[n,m] * exp(j2π(n*k/N - m*l/N))
    """
    # Build the 2D ISFFT matrix as an N² × N² operator
    n = np.arange(N)
    m = np.arange(N)
    k = np.arange(N)
    l = np.arange(N)

    # ISFFT: X_DD[n,m] → x_TF[k,l]
    # x[k,l] = 1/N * sum_{n=0}^{N-1} sum_{m=0}^{N-1} X[n,m] * exp(j2π(n*k/N - m*l/N))
    # Flatten to matrix form
    ISFFT = np.zeros((N*N, N*N), dtype=np.complex128)
    for ki in range(N):
        for li in range(N):
            for ni in range(N):
                for mi in range(N):
                    idx_in = ni * N + mi
                    idx_out = ki * N + li
                    ISFFT[idx_out, idx_in] = np.exp(1j * 2 * np.pi * (ni*ki/N - mi*li/N)) / N
    return ISFFT

def main():
    print("=" * 70)
    print("OTFS 特征直观证明 — 2D Conv 网络")
    print("=" * 70)

    from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
    import sionna.phy as sn
    sn.config.seed = SEED

    # Load trained weights
    weights_file = 'weights-qQ_Method_TV'
    print(f"\n加载权重: {weights_file}")
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)

    model = qQ_MODEL_TV(training=False)
    b, b_hat = model(2, 20.0)
    model.set_weights(weights)
    print("权重加载成功。")

    N = FFT_SIZE  # 32
    idft = compute_idft_matrix(N)
    isotfft = compute_isotfft_matrix(N)

    # ================================================================
    # 1. 提取不同 Doppler 下的 Q 矩阵
    # ================================================================
    print("\n" + "=" * 50)
    print("提取 Q 矩阵...")
    print("=" * 50)
    speeds = {'低速 (3 m/s)': 3.0, '高速 (120 m/s)': 120.0}
    q_matrices = {}

    for label, speed in speeds.items():
        Q_batch = extract_q_matrix(model, speed)
        Q_avg = np.mean(Q_batch, axis=0)
        q_matrices[label] = Q_avg
        print(f"  {label}: Q shape={Q_avg.shape}")

    Q_low = q_matrices['低速 (3 m/s)']
    Q_high = q_matrices['高速 (120 m/s)']

    # ================================================================
    # 图1：核心证明 — Q 矩阵结构 vs IDFT vs ISFFT
    # ================================================================
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    def plot_matrix(ax, matrix, title, cmap='RdBu', vrange=None):
        mag = np.abs(matrix)
        if vrange is None:
            vmax = np.max(mag)
        else:
            vmax = vrange
        im = ax.imshow(mag, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('输出维度 (时间/DD)', fontsize=9)
        ax.set_ylabel('输入维度 (频率/DD)', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: Q 矩阵幅度对比
    vmax_shared = max(np.max(np.abs(Q_low)), np.max(np.abs(Q_high)), np.max(np.abs(idft)))
    ax1 = fig.add_subplot(gs[0, 0])
    plot_matrix(ax1, Q_low, f'网络学习的 Q\n低速 3 m/s', vrange=vmax_shared)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_matrix(ax2, Q_high, f'网络学习的 Q\n高速 120 m/s', vrange=vmax_shared)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_matrix(ax3, idft, f'IDFT 矩阵 (OFDM 基准)\n|Q[i,j]|=1/√N≈{1/np.sqrt(N):.3f}', vrange=vmax_shared)

    # Plot ISFFT (reshape N²×N² to visualize first N rows as N×N)
    isotfft_N = isotfft[:N, :N]  # First N×N block
    ax4 = fig.add_subplot(gs[0, 3])
    plot_matrix(ax4, isotfft_N, f'ISFFT 矩阵 (OTFS 基准)\n前{N}×{N}子块', vrange=vmax_shared)

    # Row 2: Q 矩阵相位对比
    def plot_phase(ax, matrix, title):
        phase = np.angle(matrix)
        im = ax.imshow(phase, cmap='RdBu', aspect='auto', vmin=-np.pi, vmax=np.pi)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('输出维度', fontsize=9)
        ax.set_ylabel('输入维度', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax5 = fig.add_subplot(gs[1, 0])
    plot_phase(ax5, Q_low, 'Q 相位 — 低速')

    ax6 = fig.add_subplot(gs[1, 1])
    plot_phase(ax6, Q_high, 'Q 相位 — 高速')

    ax7 = fig.add_subplot(gs[1, 2])
    plot_phase(ax7, idft, 'IDFT 相位 — OFDM')

    ax8 = fig.add_subplot(gs[1, 3])
    plot_phase(ax8, isotfft_N, 'ISFFT 相位 — OTFS')

    # Row 3: 关键证明指标
    # 3a: Q 的各列标准差（测量扩展均匀性）
    ax9 = fig.add_subplot(gs[2, 0])
    col_std_low = np.std(np.abs(Q_low), axis=0)
    col_std_high = np.std(np.abs(Q_high), axis=0)
    col_std_idft = np.std(np.abs(idft), axis=0)
    ax9.plot(col_std_low, 'b-', alpha=0.7, label=f'低速 (mean={np.mean(col_std_low):.3f})')
    ax9.plot(col_std_high, 'r-', alpha=0.7, label=f'高速 (mean={np.mean(col_std_high):.3f})')
    ax9.axhline(y=np.mean(col_std_idft), color='gray', linestyle='--', label=f'IDFT (={np.mean(col_std_idft):.3f})')
    ax9.set_title('Q 每列 STD\n(越低=越均匀扩展=越OTFS-like)', fontsize=10)
    ax9.set_xlabel('列索引')
    ax9.set_ylabel('STD')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 3b: Q^H·Q — 正交性检验（OTFS 要求酉矩阵）
    ax10 = fig.add_subplot(gs[2, 1])
    QHQ_low = np.abs(Q_low.conj().T @ Q_low)
    np.fill_diagonal(QHQ_low, 0)  # 去掉对角线看非对角泄漏
    im10 = ax10.imshow(QHQ_low, cmap='hot', aspect='auto')
    ax10.set_title(f'Q^H·Q 非对角泄漏 — 低速\n(越暗=越正交=越OTFS)', fontsize=10)
    plt.colorbar(im10, ax=ax10, shrink=0.8)

    # 3c: Q 与 IDFT 的互相关
    ax11 = fig.add_subplot(gs[2, 2])
    # Compute normalized correlation between each column of Q and each column of IDFT
    corr_low = np.abs(Q_low.conj().T @ idft)  # N×N: Q_col_i · IDFT_col_j
    corr_high = np.abs(Q_high.conj().T @ idft)

    # Plot as histogram: how aligned is Q with IDFT?
    ax11.hist(corr_low.flatten(), bins=50, alpha=0.6, label=f'低速 (mean={np.mean(corr_low):.3f})', color='blue')
    ax11.hist(corr_high.flatten(), bins=50, alpha=0.6, label=f'高速 (mean={np.mean(corr_high):.3f})', color='red')
    ax11.axvline(x=1.0, color='green', linestyle='--', label='理想 IDFT 对齐 (=1)')
    ax11.set_title('Q 与 IDFT 列互相关分布\n(低=远离OFDM, 高=接近OFDM)', fontsize=10)
    ax11.set_xlabel('互相关幅度')
    ax11.set_ylabel('频次')
    ax11.legend(fontsize=8)

    # 3d: Q 矩阵宽度（等效扩展带宽）
    ax12 = fig.add_subplot(gs[2, 3])
    def effective_spread(Q_mat):
        """测量每行的有效扩展宽度（功率加权）"""
        mag = np.abs(Q_mat)
        N_size = mag.shape[0]
        spreads = []
        for i in range(N_size):
            row = mag[i, :]
            power = row ** 2
            total = np.sum(power)
            if total > 1e-15:
                centroid = np.sum(np.arange(N_size) * power) / total
                variance = np.sum(power * (np.arange(N_size) - centroid)**2) / total
                spreads.append(np.sqrt(variance))
            else:
                spreads.append(0)
        return np.array(spreads)

    spread_low = effective_spread(Q_low)
    spread_high = effective_spread(Q_high)
    spread_idft = effective_spread(idft)

    ax12.bar(np.arange(N)-0.2, spread_low, 0.4, alpha=0.7, label=f'低速 (mean={np.mean(spread_low):.1f})', color='blue')
    ax12.bar(np.arange(N)+0.2, spread_high, 0.4, alpha=0.7, label=f'高速 (mean={np.mean(spread_high):.1f})', color='red')
    ax12.axhline(y=np.mean(spread_idft), color='gray', linestyle='--', label=f'IDFT (={np.mean(spread_idft):.1f})')
    ax12.set_title(f'每行有效扩展宽度\n(IDFT={np.mean(spread_idft):.1f}, 越高=展得越开)', fontsize=10)
    ax12.set_xlabel('行索引')
    ax12.set_ylabel('有效扩展 (采样点)')
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    plt.suptitle('OTFS 特征证明：Q 矩阵结构分析\n'
                 f'FFT_SIZE={N} | 绿色虚线=OFDM基准 | 远离IDFT=学习非OFDM变换',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('OTFS_proof_Q_structure.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_proof_Q_structure.png")

    # ================================================================
    # 图2：单符号时频扩展响应
    # ================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (label, Q_mat) in enumerate(q_matrices.items()):
        # 发送单个符号（第 N//2 个输入=1，其余=0）
        x_input = np.zeros((1, 1, 1, 1, N), dtype=np.complex64)
        x_input[0, 0, 0, 0, N//2] = 1.0 + 0j

        # 通过 Q 变换
        Q_batch = Q_mat[np.newaxis, :, :]
        x_time = np.einsum('bxyzi,bij->bxyzj', x_input, Q_batch)[0, 0, 0, 0, :]

        # 时域响应
        ax = axes2[idx, 0]
        ax.stem(np.arange(N), np.abs(x_time), linefmt='b-', markerfmt='bo', basefmt='k-')
        ax.set_title(f'{label} — 单符号时域扩展', fontsize=12)
        ax.set_xlabel('时间采样索引')
        ax.set_ylabel('幅度')
        ax.grid(True, alpha=0.3)
        # Mark uniform spread reference
        uniform_val = 1.0 / np.sqrt(N)
        ax.axhline(y=uniform_val, color='green', linestyle='--', alpha=0.7,
                   label=f'均匀扩展 = 1/√N = {uniform_val:.3f}')
        ax.legend(fontsize=8)

        # 频域响应（通过 IDFT 看频谱）
        ax2_ = axes2[idx, 1]
        x_freq_resp = np.abs(np.fft.fft(x_time)) / np.sqrt(N)
        ax2_.stem(np.arange(N), x_freq_resp, linefmt='r-', markerfmt='ro', basefmt='k-')
        ax2_.set_title(f'{label} — 频谱响应', fontsize=12)
        ax2_.set_xlabel('子载波索引')
        ax2_.set_ylabel('幅度')
        ax2_.grid(True, alpha=0.3)

    plt.suptitle('OTFS 特征证明：单符号时频扩展\n'
                 'OTFS = 能量均匀扩展至所有时间采样 + 所有子载波 | OFDM = 集中在单一采样点',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('OTFS_proof_symbol_spread.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_proof_symbol_spread.png")

    # ================================================================
    # 图3：OFDM vs 学习到的变换 — 对比总结
    # ================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

    # 3a: 对角性度量
    def off_diagonal_ratio(matrix):
        """非对角能量占比。OFDM的IDFT ≈ 均匀，OTFS可能更高或更低"""
        mag = np.abs(matrix)
        diag = np.trace(mag)
        total = np.sum(mag)
        return 1.0 - diag / (total + 1e-10)

    matrices_to_compare = {
        '学习Q (低速)': Q_low,
        '学习Q (高速)': Q_high,
        'IDFT (OFDM)': idft,
        'ISFFT子块': isotfft_N,
    }

    names = list(matrices_to_compare.keys())
    odr_values = [off_diagonal_ratio(m) for m in matrices_to_compare.values()]
    colors = ['blue', 'red', 'gray', 'orange']
    bars = axes3[0].bar(names, odr_values, color=colors, alpha=0.7)
    axes3[0].set_title('非对角能量占比\n(越高=越非OFDM=越可能OTFS)', fontsize=11)
    axes3[0].set_ylabel('非对角能量占比')
    axes3[0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, odr_values):
        axes3[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 3b: 与 IDFT 的距离
    def frobenius_distance(A, B):
        return np.linalg.norm(A - B, 'fro') / np.linalg.norm(A, 'fro')

    dist_to_idft = [
        frobenius_distance(Q_low, idft),
        frobenius_distance(Q_high, idft),
        0.0,  # IDFT to itself
        frobenius_distance(isotfft_N, idft),
    ]
    bars2 = axes3[1].bar(names, dist_to_idft, color=colors, alpha=0.7)
    axes3[1].set_title('与 IDFT 的相对 Frobenius 距离\n(越大=越远离OFDM)', fontsize=11)
    axes3[1].set_ylabel('相对距离')
    axes3[1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars2, dist_to_idft):
        axes3[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 3c: 矩阵"扩展因子" — 每行非零元个数（以阈值计）
    def spread_factor(matrix, threshold=0.1):
        mag = np.abs(matrix)
        max_per_row = np.max(mag, axis=1, keepdims=True)
        significant = mag > threshold * max_per_row
        return np.mean(np.sum(significant, axis=1))

    spread_factors = [spread_factor(m) for m in matrices_to_compare.values()]
    bars3 = axes3[2].bar(names, spread_factors, color=colors, alpha=0.7)
    axes3[2].set_title(f'每行 >10%峰值的元素数\n(越大=扩展越广=越OTFS-like)', fontsize=11)
    axes3[2].set_ylabel('有效扩展元素数')
    axes3[2].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars3, spread_factors):
        axes3[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                     f'{val:.1f}/{N}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('OTFS vs OFDM：变换矩阵多维度对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('OTFS_proof_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_proof_comparison.png")

    # ================================================================
    # 最终总结
    # ================================================================
    print("\n" + "=" * 70)
    print("OTFS 特征直观证明 — 总结")
    print("=" * 70)

    odr_low = off_diagonal_ratio(Q_low)
    odr_high = off_diagonal_ratio(Q_high)
    odr_idft = off_diagonal_ratio(idft)
    odr_isotfft = off_diagonal_ratio(isotfft_N)

    dist_low = frobenius_distance(Q_low, idft)
    dist_high = frobenius_distance(Q_high, idft)

    sf_low = spread_factor(Q_low)
    sf_high = spread_factor(Q_high)
    sf_idft = spread_factor(idft)

    # Q^H·Q orthogonality check
    QHQ_low_full = Q_low.conj().T @ Q_low
    QHQ_high_full = Q_high.conj().T @ Q_high
    ortho_low = np.max(np.abs(QHQ_low_full - np.eye(N)))
    ortho_high = np.max(np.abs(QHQ_high_full - np.eye(N)))
    ortho_idft = np.max(np.abs(idft.conj().T @ idft - np.eye(N)))

    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │                 OTFS 特征证明 — 量化指标                  │
    ├──────────────────┬──────────┬──────────┬───────┬────────┤
    │ 指标              │ 低速 Q   │ 高速 Q   │ IDFT  │ ISFFT  │
    ├──────────────────┼──────────┼──────────┼───────┼────────┤
    │ 非对角能量占比     │ {odr_low:.4f}  │ {odr_high:.4f}  │ {odr_idft:.4f} │ {odr_isotfft:.4f} │
    │ 与 IDFT 距离       │ {dist_low:.4f}  │ {dist_high:.4f}  │ 0.0   │ {frobenius_distance(isotfft_N, idft):.4f} │
    │ 扩展因子 (>10%)    │ {sf_low:.1f}/{N}   │ {sf_high:.1f}/{N}   │ {sf_idft:.1f}/{N}  │ {spread_factor(isotfft_N):.1f}/{N} │
    │ Q^H·Q 偏离 I 度    │ {ortho_low:.4f}  │ {ortho_high:.4f}  │ {ortho_idft:.4f} │ N/A    │
    └──────────────────┴──────────┴──────────┴───────┴────────┘
    """)

    # 判断逻辑（类似 OFDM 的"看频谱"）
    print('  判定逻辑（对应 OFDM 的"看基底/看频谱"）：')
    print()

    checks = []

    # Check 1: Non-diagonal energy
    if odr_low > odr_idft * 1.1 and odr_high > odr_idft * 1.1:
        checks.append(('✅ 通过', f'Q 非对角能量 ({odr_low:.3f}/{odr_high:.3f}) > IDFT ({odr_idft:.3f}) → 非纯OFDM'))
    else:
        checks.append(('❌ 未通过', f'Q 过于接近对角 → 可能退化到 OFDM'))

    # Check 2: Distance from IDFT
    if dist_low > 0.1 and dist_high > 0.1:
        checks.append(('✅ 通过', f'Q 与 IDFT 距离显著 ({dist_low:.3f}/{dist_high:.3f}) → 学习了不同变换'))
    else:
        checks.append(('❌ 未通过', f'Q 与 IDFT 距离过小 → 近似 OFDM'))

    # Check 3: Spread factor
    if sf_low > sf_idft and sf_high > sf_idft:
        checks.append(('✅ 通过', f'Q 扩展因子 ({sf_low:.1f}/{sf_high:.1f}) > IDFT ({sf_idft:.1f}) → 有扩展特性'))
    else:
        checks.append(('⚠️  部分', f'扩展因子未显著超过 IDFT'))

    # Check 4: Doppler adaptation
    q_diff = np.mean(np.abs(Q_high - Q_low))
    if q_diff > 0.01:
        checks.append(('✅ 通过', f'Q 随 Doppler 变化 (平均差异={q_diff:.4f}) → 自适应'))
    else:
        checks.append(('❌ 未通过', f'Q 不随 Doppler 变化 → 非自适应'))

    # Check 5: Approximate unitarity
    if ortho_low < 0.5 and ortho_high < 0.5:
        checks.append(('✅ 通过', f'Q 近似酉矩阵 (偏离={ortho_low:.3f}/{ortho_high:.3f}) → 能量守恒'))
    else:
        checks.append(('⚠️  部分', f'Q 偏离酉性较大'))

    for status, msg in checks:
        print(f"  {status}: {msg}")

    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │ 直观类比：                                               │
    │                                                         │
    │ OFDM 证明 = 看基底正弦波 + 看多载波频谱                    │
    │ OTFS 证明 = 看 Q 非对角 + 看单符号扩展 + 看酉性           │
    │                                                         │
    │ 如果 Q ≈ IDFT → 网络"退化"为 OFDM                        │
    │ 如果 Q 是非对角扩展矩阵且酉 → 网络学到了"类 OTFS"变换      │
    └─────────────────────────────────────────────────────────┘
    """)

    print("可视化完成。生成文件：")
    print("  OTFS_proof_Q_structure.png   — Q 矩阵结构与 IDFT/ISFFT 对比")
    print("  OTFS_proof_symbol_spread.png — 单符号时频扩展响应")
    print("  OTFS_proof_comparison.png    — 多维度量化对比")

if __name__ == "__main__":
    main()
