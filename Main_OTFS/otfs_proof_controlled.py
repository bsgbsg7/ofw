#!/usr/bin/env python3
"""
OTFS 特征证明 — 控制变量实验
============================
核心问题：延迟扩展(delay spread)和 Doppler 同时随机变化，
导致无法区分 Q 的变化是来自时延还是多普勒。

控制变量方案：
  - 固定延迟扩展 (100ns = "Nominal delay spread")
  - 仅变速度：3 m/s (低 Doppler) vs 120 m/s (高 Doppler)
  - 目标：低速 → Q ≈ IDFT (类 OFDM)，高速 → Q 是非平凡扩展矩阵 (类 OTFS)
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

def build_fixed_ds_channel(speed_min, speed_max, delay_spread=100e-9):
    """
    构建固定延迟扩展 + 指定速度范围的信道。
    延迟扩展固定在 delay_spread，只有速度变化。
    +1e-9 是因为 TDL_RandomDS 内部使用整数纳秒采样，要求 min < max。
    """
    from utils.TDL_RandomDS import TDL_RandomDS
    return TDL_RandomDS(
        model='A',
        delay_spread_min=delay_spread,
        delay_spread_max=delay_spread + 1e-9,  # +1ns 绕过 min<max 限制
        carrier_frequency=CARRIER_FREQ,
        min_speed=speed_min,
        max_speed=speed_max
    )

def extract_q_controlled(model, speed, delay_spread=100e-9, batch_size=16):
    """提取固定延迟扩展下的 Q 矩阵。"""
    from sionna.phy.channel import cir_to_time_channel
    original_channel = model._channel_model

    # 窄速度范围，固定延迟扩展
    test_channel = build_fixed_ds_channel(
        max(0.1, speed - 0.5), speed + 0.5, delay_spread
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
    pilots = tf.transpose(pilots, [0, 2, 1])

    creator = model._qQ_creator_layer
    Q, _ = creator(pilots, training=False)
    model._channel_model = original_channel
    return Q.numpy()

def compute_idft_matrix(N):
    k = np.arange(N)
    n = np.arange(N).reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def main():
    print("=" * 70)
    print("OTFS 特征证明 — 控制变量实验 (固定延迟扩展, 仅变 Doppler)")
    print("=" * 70)

    from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
    import sionna.phy as sn
    sn.config.seed = SEED

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

    # ================================================================
    # 控制变量实验：三种延迟扩展 × 两种速度
    # ================================================================
    delay_spreads = {
        '短时延 (50ns)': 50e-9,
        '中等时延 (100ns)': 100e-9,
        '长时延 (300ns)': 300e-9,
    }
    speeds = {
        '低速 3 m/s': 3.0,
        '中速 60 m/s': 60.0,
        '高速 120 m/s': 120.0,
    }

    print("\n提取 Q 矩阵 (固定延迟扩展, 仅变速度)...")
    all_q = {}  # key: (ds_label, speed_label) -> Q_avg
    for ds_label, ds_val in delay_spreads.items():
        for sp_label, sp_val in speeds.items():
            Q_batch = extract_q_controlled(model, sp_val, ds_val, batch_size=8)
            Q_avg = np.mean(Q_batch, axis=0)
            all_q[(ds_label, sp_label)] = Q_avg
            dist_to_idft = np.linalg.norm(Q_avg - idft, 'fro') / np.linalg.norm(Q_avg, 'fro')
            print(f"  {ds_label} + {sp_label}: |Q|=({np.min(np.abs(Q_avg)):.3f},{np.max(np.abs(Q_avg)):.3f}), 与IDFT距离={dist_to_idft:.3f}")

    # ================================================================
    # 图1：核心 — 固定延迟 100ns，速度 3 vs 60 vs 120 m/s 的 Q 对比
    # ================================================================
    ds_key = '中等时延 (100ns)'
    q_low = all_q[(ds_key, '低速 3 m/s')]
    q_mid = all_q[(ds_key, '中速 60 m/s')]
    q_high = all_q[(ds_key, '高速 120 m/s')]

    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.35)

    def plot_matrix(ax, matrix, title, cmap='hot'):
        mag = np.abs(matrix)
        im = ax.imshow(mag, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('输出 (时间采样)', fontsize=9)
        ax.set_ylabel('输入 (频率)', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 0: 幅度对比 — IDFT, Q@3m/s, Q@60m/s, Q@120m/s, Q_diff(120-3)
    vmax_shared = max(np.max(np.abs(q_low)), np.max(np.abs(q_high)), np.max(np.abs(idft)))
    ax = fig.add_subplot(gs[0, 0])
    plot_matrix(ax, idft, 'IDFT (OFDM 基准)\n每元素幅度=%.3f' % (1/np.sqrt(N)))

    ax = fig.add_subplot(gs[0, 1])
    plot_matrix(ax, q_low, f'Q — 低速 3 m/s\n与IDFT距离={np.linalg.norm(q_low-idft,"fro")/np.linalg.norm(q_low,"fro"):.3f}')

    ax = fig.add_subplot(gs[0, 2])
    plot_matrix(ax, q_mid, f'Q — 中速 60 m/s\n与IDFT距离={np.linalg.norm(q_mid-idft,"fro")/np.linalg.norm(q_mid,"fro"):.3f}')

    ax = fig.add_subplot(gs[0, 3])
    plot_matrix(ax, q_high, f'Q — 高速 120 m/s\n与IDFT距离={np.linalg.norm(q_high-idft,"fro")/np.linalg.norm(q_high,"fro"):.3f}')

    # Q 差分图 (高速 - 低速)
    ax = fig.add_subplot(gs[0, 4])
    q_diff = q_high - q_low
    diff_mag = np.abs(q_diff)
    im = ax.imshow(diff_mag, cmap='RdBu', aspect='auto')
    ax.set_title(f'Q 差分: 120m/s − 3m/s\nmax|diff|={np.max(diff_mag):.4f}, mean|diff|={np.mean(diff_mag):.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('输出 (时间采样)', fontsize=9)
    ax.set_ylabel('输入 (频率)', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: 相位对比
    def plot_phase(ax, matrix, title):
        phase = np.angle(matrix)
        im = ax.imshow(phase, cmap='RdBu', aspect='auto', vmin=-np.pi, vmax=np.pi)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[1, 0])
    plot_phase(ax, idft, 'IDFT 相位')

    ax = fig.add_subplot(gs[1, 1])
    plot_phase(ax, q_low, 'Q 相位 — 3 m/s')

    ax = fig.add_subplot(gs[1, 2])
    plot_phase(ax, q_mid, 'Q 相位 — 60 m/s')

    ax = fig.add_subplot(gs[1, 3])
    plot_phase(ax, q_high, 'Q 相位 — 120 m/s')

    ax = fig.add_subplot(gs[1, 4])
    phase_diff = np.angle(q_high) - np.angle(q_low)
    im = ax.imshow(phase_diff, cmap='RdBu', aspect='auto', vmin=-np.pi, vmax=np.pi)
    ax.set_title('相位差: 120m/s − 3m/s', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2: 单符号时域扩展 + 频谱
    for idx, (label, Q_mat) in enumerate([
        ('IDFT (OFDM)', idft),
        ('Q 低速 3 m/s', q_low),
        ('Q 中速 60 m/s', q_mid),
        ('Q 高速 120 m/s', q_high),
    ]):
        # 时域扩展
        ax_t = fig.add_subplot(gs[2, idx])
        x_input = np.zeros((1, 1, 1, 1, N), dtype=np.complex64)
        x_input[0, 0, 0, 0, N//2] = 1.0 + 0j
        Q_batch = Q_mat[np.newaxis, :, :]
        x_time = np.einsum('bxyzi,bij->bxyzj', x_input, Q_batch)[0, 0, 0, 0, :]

        ax_t.stem(np.arange(N), np.abs(x_time), linefmt='b-', markerfmt='bo', basefmt='k-')
        uniform_val = 1.0 / np.sqrt(N)
        ax_t.axhline(y=uniform_val, color='green', linestyle='--', alpha=0.5, label=f'均匀=1/√N={uniform_val:.3f}')
        spread_std = np.std(np.abs(x_time))
        ax_t.set_title(f'{label}\n扩展STD={spread_std:.4f}', fontsize=10)
        ax_t.set_xlabel('时间采样', fontsize=9)
        ax_t.set_ylabel('幅度', fontsize=9)
        ax_t.legend(fontsize=7)
        ax_t.grid(True, alpha=0.3)

    # 右下角：与 IDFT 距离随速度变化汇总
    ax_sum = fig.add_subplot(gs[2, 4])
    for ds_label in delay_spreads:
        distances = []
        speed_vals = []
        for sp_label, sp_val in speeds.items():
            Q_mat = all_q[(ds_label, sp_label)]
            d = np.linalg.norm(Q_mat - idft, 'fro') / np.linalg.norm(Q_mat, 'fro')
            distances.append(d)
            speed_vals.append(sp_val)
        ax_sum.plot(speed_vals, distances, 'o-', linewidth=2, markersize=8, label=ds_label)
    ax_sum.set_title('与 IDFT 距离 vs 速度\n(越大=越远离OFDM)', fontsize=11)
    ax_sum.set_xlabel('速度 [m/s]', fontsize=10)
    ax_sum.set_ylabel('Frobenius 相对距离', fontsize=10)
    ax_sum.legend(fontsize=8)
    ax_sum.grid(True, alpha=0.3)

    plt.suptitle(f'OTFS 特征证明 — 控制变量 (固定延迟扩展, 仅变 Doppler)\n'
                 f'FFT_SIZE={N} | 延迟扩展固定 100ns | 绿色虚线=均匀扩展基准',
                 fontsize=14, fontweight='bold')
    plt.savefig('OTFS_proof_controlled.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_proof_controlled.png")

    # ================================================================
    # 图2：逐列分析 — 哪些频率分量在 Doppler 下变化最大？
    # ================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

    # 2a: 每列与 IDFT 对应列的差异 (高速 vs 低速)
    ax = axes2[0, 0]
    for sp_label in speeds:
        Q_mat = all_q[(ds_key, sp_label)]
        col_dists = [np.linalg.norm(Q_mat[:, i] - idft[:, i]) for i in range(N)]
        ax.plot(col_dists, 'o-', label=sp_label, linewidth=1.5, markersize=4)
    ax.set_title('每列与 IDFT 对应列的距离\n(越大=该频率分量变换越非OFDM)', fontsize=10)
    ax.set_xlabel('列索引 (频率分量)')
    ax.set_ylabel('L2 距离')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2b: Q 每列的"时域扩展宽度" (功率加权标准差)
    ax = axes2[0, 1]
    def col_time_spread(matrix):
        mag = np.abs(matrix)
        spreads = []
        for i in range(N):
            col = mag[:, i]
            power = col ** 2
            total = np.sum(power)
            if total > 1e-15:
                centroid = np.sum(np.arange(N) * power) / total
                variance = np.sum(power * (np.arange(N) - centroid) ** 2) / total
                spreads.append(np.sqrt(variance))
            else:
                spreads.append(0)
        return np.array(spreads)

    idft_spread = col_time_spread(idft)
    ax.axhline(y=np.mean(idft_spread), color='green', linestyle='--', label=f'IDFT mean={np.mean(idft_spread):.1f}')
    for sp_label in speeds:
        Q_mat = all_q[(ds_key, sp_label)]
        s = col_time_spread(Q_mat)
        ax.plot(s, 'o-', label=f'{sp_label} mean={np.mean(s):.1f}', linewidth=1.5, markersize=4)
    ax.set_title('每列的时域扩展宽度\n(IDFT=均匀扩展, 偏离=非OFDM结构)', fontsize=10)
    ax.set_xlabel('列索引 (频率分量)')
    ax.set_ylabel('扩展宽度 (采样点)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2c: Q^H·Q 偏离单位阵 — 正交性检验
    ax = axes2[0, 2]
    ortho_vals = []
    for sp_label in speeds:
        Q_mat = all_q[(ds_key, sp_label)]
        QHQ = Q_mat.conj().T @ Q_mat
        off_diag = np.abs(QHQ - np.eye(N))
        ortho_vals.append(np.max(off_diag))
    ax.bar([s.replace(' m/s', '') for s in speeds.keys()], ortho_vals, color=['blue', 'orange', 'red'], alpha=0.7)
    ax.axhline(y=0, color='green', linestyle='--', label='理想酉矩阵 = 0')
    ax.set_title('Q^H·Q 偏离单位阵\n(0=完美正交=纯OFDM/OTFS)', fontsize=10)
    ax.set_ylabel('最大非对角幅度')
    ax.legend(fontsize=8)

    # 2d: 单符号响应 — 时域包络比较 (叠加)
    ax = axes2[1, 0]
    x_input = np.zeros((1, 1, 1, 1, N), dtype=np.complex64)
    x_input[0, 0, 0, 0, N//2] = 1.0 + 0j
    for sp_label, color in zip(speeds.keys(), ['blue', 'orange', 'red']):
        Q_mat = all_q[(ds_key, sp_label)]
        x_time = np.einsum('bxyzi,bij->bxyzj', x_input, Q_mat[np.newaxis,:,:])[0,0,0,0,:]
        ax.plot(np.abs(x_time), color=color, linewidth=1.5, alpha=0.8,
                marker='o', markersize=3, label=sp_label)
    idft_time = np.einsum('bxyzi,bij->bxyzj', x_input, idft[np.newaxis,:,:])[0,0,0,0,:]
    ax.plot(np.abs(idft_time), 'g--', linewidth=1, alpha=0.5, label='IDFT (OFDM)')
    ax.set_title('单符号时域包络叠加\n(相同色=无Doppler自适应)', fontsize=10)
    ax.set_xlabel('时间采样')
    ax.set_ylabel('幅度')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2e: 频域响应比较
    ax = axes2[1, 1]
    for sp_label, color in zip(speeds.keys(), ['blue', 'orange', 'red']):
        Q_mat = all_q[(ds_key, sp_label)]
        x_time = np.einsum('bxyzi,bij->bxyzj', x_input, Q_mat[np.newaxis,:,:])[0,0,0,0,:]
        x_freq = np.abs(np.fft.fft(x_time)) / np.sqrt(N)
        ax.plot(x_freq, color=color, linewidth=1.5, alpha=0.8,
                marker='s', markersize=3, label=sp_label)
    idft_freq = np.abs(np.fft.fft(idft_time)) / np.sqrt(N)
    ax.plot(idft_freq, 'g--', linewidth=1, alpha=0.5, label='IDFT (OFDM)')
    ax.set_title('单符号频谱响应\n(OFDM=单一子载波峰值)', fontsize=10)
    ax.set_xlabel('子载波索引')
    ax.set_ylabel('幅度')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2f: 各延迟扩展下，高速/低速的 Q 差分幅度
    ax = axes2[1, 2]
    x_pos = np.arange(len(delay_spreads))
    width = 0.25
    for i, (sp_label, color) in enumerate(zip(speeds.keys(), ['blue', 'orange', 'red'])):
        diffs = []
        for ds_label in delay_spreads:
            # 相对于 IDFT 的距离
            Q_mat = all_q[(ds_label, sp_label)]
            d = np.linalg.norm(Q_mat - idft, 'fro') / np.linalg.norm(Q_mat, 'fro')
            diffs.append(d)
        ax.bar(x_pos + i*width, diffs, width, alpha=0.7, color=color, label=sp_label)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([s.split('(')[0] for s in delay_spreads.keys()])
    ax.set_title('不同延迟扩展下 Q 与 IDFT 的距离', fontsize=10)
    ax.set_ylabel('Frobenius 相对距离')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('OTFS 特征证明 — 控制变量详细分析\n'
                 f'FFT_SIZE={N} | 延迟扩展固定 | 仅 Doppler 变化',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('OTFS_proof_controlled_detail.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: OTFS_proof_controlled_detail.png")

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 70)
    print("控制变量实验 — 结论")
    print("=" * 70)

    print(f"\n{'延迟扩展':<20} {'速度':<15} {'与IDFT距离':<12} {'Q^H·Q偏离I':<12} {'幅度STD':<10}")
    print("-" * 70)
    for ds_label in delay_spreads:
        for sp_label in speeds:
            Q_mat = all_q[(ds_label, sp_label)]
            dist = np.linalg.norm(Q_mat - idft, 'fro') / np.linalg.norm(Q_mat, 'fro')
            QHQ = Q_mat.conj().T @ Q_mat
            ortho = np.max(np.abs(QHQ - np.eye(N)))
            mag_std = np.std(np.abs(Q_mat))
            print(f"{ds_label:<20} {sp_label:<15} {dist:<12.4f} {ortho:<12.4f} {mag_std:<10.4f}")

    # 关键判定
    q_low_dist = np.linalg.norm(q_low - idft, 'fro') / np.linalg.norm(q_low, 'fro')
    q_high_dist = np.linalg.norm(q_high - idft, 'fro') / np.linalg.norm(q_high, 'fro')
    q_diff_mag = np.mean(np.abs(q_high - q_low))

    print(f"""
    ┌──────────────────────────────────────────────────────────────┐
    │  控制变量实验 (延迟扩展固定 100ns, 仅 Doppler 变化)           │
    ├──────────────────────────────────────────────────────────────┤
    │  低速 (3 m/s) 与 IDFT 距离:    {q_low_dist:.4f}                         │
    │  高速 (120 m/s) 与 IDFT 距离:  {q_high_dist:.4f}                         │
    │  高低速 Q 平均差异:            {q_diff_mag:.4f}                         │
    │                                                              │
    │  判定逻辑:                                                    │
    │  · 若低速距离 < 高速距离 → 网络在高速下更偏离OFDM → OTFS-like │
    │  · 若高低速 Q 有显著差异 → 网络确实在自适应 Doppler        │
    └──────────────────────────────────────────────────────────────┘
    """)

    if q_low_dist < q_high_dist:
        print("  ✅ 低速更接近 IDFT，高速更偏离 IDFT → 网络在高速下学习非 OFDM 变换")
    else:
        print("  ⚠️ 低速与高速的 IDFT 距离相近 → 变换与 Doppler 的关系不明显")

    if q_diff_mag > 0.005:
        print(f"  ✅ Q 随 Doppler 显著变化 (|ΔQ|={q_diff_mag:.4f}) → 网络有 Doppler 自适应")
    else:
        print(f"  ❌ Q 几乎不随 Doppler 变化 (|ΔQ|={q_diff_mag:.4f}) → 网络无 Doppler 自适应")

    print("\n可视化完成。生成文件：")
    print("  OTFS_proof_controlled.png        — 控制变量 Q 矩阵对比")
    print("  OTFS_proof_controlled_detail.png — 详细分析 (逐列/正交性/频谱)")

if __name__ == "__main__":
    main()
