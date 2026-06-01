#!/usr/bin/env python3
"""
验证脚本: 测试网络在三种信道条件下是否产生不同波形
=====================================================
场景 1: 平坦信道 + 低速 → 预期 Q ≈ I (TDM/单载波)
场景 2: 多径信道 + 低速 → 预期 Q ≈ IDFT (OFDM)
场景 3: 多径信道 + 高速 → 预期 Q 是密集非 IDFT 矩阵 (OTFS)
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

def build_channel(speed_min, speed_max, ds_min, ds_max):
    from utils.TDL_RandomDS import TDL_RandomDS
    return TDL_RandomDS(
        model='A',
        delay_spread_min=ds_min,
        delay_spread_max=ds_max,
        carrier_frequency=CARRIER_FREQ,
        min_speed=speed_min,
        max_speed=speed_max
    )

def extract_q(model, speed, ds_min, ds_max, batch_size=16):
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
    pilots = tf.transpose(pilots, [0, 2, 1])
    creator = model._qQ_creator_layer
    Q, _ = creator(pilots, training=False)
    model._channel_model = original_channel
    return Q.numpy()

def compute_idft_matrix(N):
    k = np.arange(N); n = np.arange(N).reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def main():
    print("=" * 64)
    print("TDM / OFDM / OTFS 波形验证")
    print("=" * 64)

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

    N = FFT_SIZE
    idft = compute_idft_matrix(N)
    eye = np.eye(N, dtype=np.complex128) / np.sqrt(N)  # normalized identity

    # ================================================================
    # 三种测试场景
    # ================================================================
    scenarios = [
        # (名称, 速度, ds_min, ds_max, 预期)
        ('场景1: 平坦+低速\n预期 TDM',   3.0,  10e-9, 11e-9),
        ('场景1b: 平坦+高速\n预期 TDM',  120.0, 10e-9, 11e-9),
        ('场景2: 多径+低速\n预期 OFDM',  3.0,  100e-9, 101e-9),
        ('场景2b: 多径+中速\n预期 OFDM', 60.0, 100e-9, 101e-9),
        ('场景3: 多径+高速\n预期 OTFS',  120.0, 300e-9, 301e-9),
        ('场景3b: 多径+超高速',          200.0, 300e-9, 301e-9),
    ]

    q_matrices = {}
    for name, speed, ds_min, ds_max in scenarios:
        Q_batch = extract_q(model, speed, ds_min, ds_max, batch_size=8)
        Q_avg = np.mean(Q_batch, axis=0)
        q_matrices[name] = Q_avg
        d_idft = np.linalg.norm(Q_avg - idft, 'fro') / np.linalg.norm(Q_avg, 'fro')
        d_eye = np.linalg.norm(Q_avg - eye, 'fro') / np.linalg.norm(Q_avg, 'fro')
        QHQ = Q_avg.conj().T @ Q_avg
        ortho_err = np.max(np.abs(QHQ - np.eye(N)))
        print(f"  {name.split(chr(10))[0]}: dist_to_IDFT={d_idft:.3f}, dist_to_I={d_eye:.3f}, ortho_err={ortho_err:.3f}")

    # ================================================================
    # 大图: 3×4 网格对比
    # ================================================================
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 8, figure=fig, hspace=0.4, wspace=0.35)

    def plot_matrix(ax, matrix, title, cmap='hot'):
        mag = np.abs(matrix)
        im = ax.imshow(mag, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 0: 幅度
    for idx, (name, speed, ds_min, ds_max) in enumerate(scenarios):
        if idx >= 6:
            break
        ax = fig.add_subplot(gs[0, idx])
        Q = q_matrices[name]
        d_idft = np.linalg.norm(Q - idft, 'fro') / np.linalg.norm(Q, 'fro')
        d_eye = np.linalg.norm(Q - eye, 'fro') / np.linalg.norm(Q, 'fro')
        plot_matrix(ax, Q, f'{name}\nd_IDFT={d_idft:.3f} d_I={d_eye:.3f}')

    # IDFT reference
    ax = fig.add_subplot(gs[0, 6])
    plot_matrix(ax, idft, 'IDFT (OFDM基准)')
    # Identity reference
    ax = fig.add_subplot(gs[0, 7])
    plot_matrix(ax, eye, 'I (TDM基准)')

    # Row 1: Q 与 IDFT 的差值幅度 |Q - IDFT|
    for idx, (name, speed, ds_min, ds_max) in enumerate(scenarios):
        if idx >= 6:
            break
        ax = fig.add_subplot(gs[1, idx])
        Q = q_matrices[name]
        diff = np.abs(Q - idft)
        im = ax.imshow(diff, cmap='Reds', aspect='auto')
        ax.set_title(f'|Q − IDFT| — {name.split(chr(10))[0]}', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # IDFT vs IDFT (should be zero)
    ax = fig.add_subplot(gs[1, 6])
    im = ax.imshow(np.abs(idft - idft), cmap='Reds', aspect='auto')
    ax.set_title('|IDFT − IDFT| = 0', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # I vs IDFT difference
    ax = fig.add_subplot(gs[1, 7])
    im = ax.imshow(np.abs(eye - idft), cmap='Reds', aspect='auto')
    ax.set_title('|I − IDFT| (TDM vs OFDM)', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2: 单符号时域扩展
    for idx, (name, speed, ds_min, ds_max) in enumerate(scenarios):
        if idx >= 6:
            break
        ax = fig.add_subplot(gs[2, idx])
        Q = q_matrices[name]
        x_input = np.zeros((1, 1, 1, 1, N), dtype=np.complex64)
        x_input[0, 0, 0, 0, N//2] = 1.0 + 0j
        x_time = np.einsum('bxyzi,bij->bxyzj', x_input, Q[np.newaxis,:,:])[0,0,0,0,:]
        mag = np.abs(x_time)
        ax.stem(range(N), mag, linefmt='b-', markerfmt='bo', basefmt='k-')
        uniform_val = 1.0 / np.sqrt(N)
        ax.axhline(y=uniform_val, color='green', linestyle='--', alpha=0.5, label=f'uniform={uniform_val:.3f}')
        ax.set_title(f'{name.split(chr(10))[0]}\nSTD={np.std(mag):.4f}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # IDFT single symbol response
    ax = fig.add_subplot(gs[2, 6])
    idft_time = np.einsum('bxyzi,bij->bxyzj', x_input, idft[np.newaxis,:,:])[0,0,0,0,:]
    ax.stem(range(N), np.abs(idft_time), linefmt='g-', markerfmt='go', basefmt='k-')
    ax.set_title('IDFT (OFDM)\n完全均匀', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Identity single symbol response
    ax = fig.add_subplot(gs[2, 7])
    eye_time = np.einsum('bxyzi,bij->bxyzj', x_input, eye[np.newaxis,:,:])[0,0,0,0,:]
    ax.stem(range(N), np.abs(eye_time), linefmt='r-', markerfmt='ro', basefmt='k-')
    ax.set_title('I (TDM)\n单一峰值', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('TDM / OFDM / OTFS 波形验证\n'
                 '平坦+低速→TDM(I) | 多径+低速→OFDM(IDFT) | 多径+高速→OTFS(非I非IDFT)',
                 fontsize=14, fontweight='bold')
    plt.savefig('verify_tdm_ofdm_otfs.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: verify_tdm_ofdm_otfs.png")

    # ================================================================
    # 总结指标
    # ================================================================
    print("\n" + "=" * 64)
    print("验证总结")
    print("=" * 64)
    print(f"\n{'场景':<30} {'dist_IDFT':<12} {'dist_I':<12} {'判定':<30}")
    print("-" * 70)

    for name, _, _, _ in scenarios:
        Q = q_matrices[name]
        d_idft = np.linalg.norm(Q - idft, 'fro') / np.linalg.norm(Q, 'fro')
        d_eye = np.linalg.norm(Q - eye, 'fro') / np.linalg.norm(Q, 'fro')

        if '平坦' in name:
            # Expect close to I (TDM)
            verdict = '✅ TDM' if d_eye < d_idft else '⚠️ 非TDM'
        elif '低速' in name or '中速' in name:
            # Expect close to IDFT (OFDM)
            verdict = '✅ OFDM' if d_idft < d_eye and d_idft < 1.0 else '⚠️ 非OFDM'
        elif '高速' in name or '超高速' in name:
            # Expect far from both I and IDFT (OTFS)
            verdict = '✅ OTFS' if d_idft > 0.5 and d_eye > 0.5 else '⚠️ 非OTFS'
        else:
            verdict = '—'

        print(f"{name.split(chr(10))[0]:<30} {d_idft:<12.4f} {d_eye:<12.4f} {verdict:<30}")

    print(f"\n判定规则:")
    print(f"  TDM:  dist_I < dist_IDFT (接近单位阵)")
    print(f"  OFDM: dist_IDFT < dist_I 且 dist_IDFT < 1.0 (接近IDFT)")
    print(f"  OTFS: dist_IDFT > 0.5 且 dist_I > 0.5 (远离两者)")

if __name__ == "__main__":
    main()
