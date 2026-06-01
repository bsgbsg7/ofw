#!/usr/bin/env python3
"""
可视化：架构图 + CIR对比 + Q基底对比
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
import sionna.phy as sn
from sionna.phy.channel import cir_to_time_channel

sn.config.seed = SEED

# ================================================================
# 1. 加载模型
# ================================================================
from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
from utils.TDL_RandomDS import TDL_RandomDS

model = qQ_MODEL_TV(training=False)
model(2, 40.0)

weights_file = 'weights-qQ_Method_TV'
print(f"Loading weights from {weights_file}")
with open(weights_file, 'rb') as f:
    model.set_weights(pickle.load(f))
print("Weights loaded.")

# ================================================================
# 2. 构建信道 + 提取 CIR 和 Q
# ================================================================
def extract_cir_and_q(model, speed, ds_min, ds_max, label=""):
    """Extract CIR, Q matrix for a specific channel condition."""
    from sionna.phy.channel import cir_to_time_channel

    # Build fixed channel
    test_ch = TDL_RandomDS(
        model='A',
        delay_spread_min=ds_min,
        delay_spread_max=ds_max + 1e-9,
        carrier_frequency=CARRIER_FREQ,
        min_speed=max(0.1, speed - 0.5),
        max_speed=speed + 0.5
    )

    original_ch = model._channel_model
    model._channel_model = test_ch

    a, tau = test_ch(8, model._rg.num_time_samples + model._l_tot - 1, model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                  l_min=model._l_min, l_max=model._l_max, normalize=True)

    h_2d = h_time[:, 0, 0, 0, 0, :, :]
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    cir_snapshots = tf.gather(h_2d, indices, axis=1)  # (batch, num_snapshots, l_tot)
    cir_input = tf.transpose(cir_snapshots, [0, 2, 1])  # (batch, l_tot, num_snapshots)

    Q, _ = model._qQ_creator_layer(cir_input, training=False)

    model._channel_model = original_ch
    return cir_input.numpy(), Q.numpy()

def compute_idft_matrix(N):
    k = np.arange(N); n = np.arange(N).reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * k * n / N) / np.sqrt(N)

N = FFT_SIZE
idft = compute_idft_matrix(N)
eye = np.eye(N, dtype=np.complex128) / np.sqrt(N)

# 两个场景
scenarios = [
    ("平坦 3m/s (目标TDM)",  3.0,  10e-9, 11e-9),
    ("多径 3m/s (目标OFDM)", 3.0,  200e-9, 201e-9),
]

results = {}
for name, speed, ds_min, ds_max in scenarios:
    cir, Q_batch = extract_cir_and_q(model, speed, ds_min, ds_max)
    Q_avg = np.mean(Q_batch, axis=0)
    results[name] = {'cir': cir[0], 'Q': Q_avg}  # first sample

    d_idft = np.linalg.norm(Q_avg - idft, 'fro') / np.linalg.norm(Q_avg, 'fro')
    d_eye  = np.linalg.norm(Q_avg - eye, 'fro') / np.linalg.norm(Q_avg, 'fro')
    print(f"  {name}: dist_IDFT={d_idft:.4f}, dist_I={d_eye:.4f}")

# ================================================================
# 3. 大图
# ================================================================
fig = plt.figure(figsize=(22, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# ---- Row 0: CIR (delay × time) ----
for idx, (name, _, _, _) in enumerate(scenarios):
    ax = fig.add_subplot(gs[0, idx * 2])
    cir = results[name]['cir']  # (l_tot, num_snapshots)
    cir_mag = np.abs(cir)
    im = ax.imshow(cir_mag, aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('Time snapshot', fontsize=10)
    ax.set_ylabel('Delay tap', fontsize=10)
    ax.set_title(f'CIR: {name.split(chr(10))[0]}\nShape: {cir.shape[0]} taps × {cir.shape[1]} snapshots',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

# CIR 时间演化 (选 3 条 delay tap 看时间变化)
for idx, (name, _, _, _) in enumerate(scenarios):
    ax = fig.add_subplot(gs[0, idx * 2 + 1])
    cir = results[name]['cir']
    cir_mag = np.abs(cir)
    l_tot = cir.shape[0]
    # Pick taps with most energy
    tap_energy = np.sum(cir_mag, axis=1)
    top_taps = np.argsort(tap_energy)[-4:]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for i, tap in enumerate(top_taps):
        ax.plot(cir_mag[tap, :], color=colors[i], linewidth=1.5,
                label=f'tap {tap}', marker='o', markersize=3)
    ax.set_xlabel('Time snapshot', fontsize=10)
    ax.set_ylabel('|CIR|', fontsize=10)
    ax.set_title(f'CIR time evolution: {name.split(chr(10))[0]}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ---- Row 1: Q 矩阵幅度 + 基底 (前8行) ----
for idx, (name, _, _, _) in enumerate(scenarios):
    # Q magnitude
    ax1 = fig.add_subplot(gs[1, idx * 2])
    Q = results[name]['Q']
    im = ax1.imshow(np.abs(Q), aspect='auto', cmap='hot')
    ax1.set_xlabel('Output index', fontsize=10)
    ax1.set_ylabel('Input index', fontsize=10)
    d_idft = np.linalg.norm(Q - idft, 'fro') / np.linalg.norm(Q, 'fro')
    d_eye = np.linalg.norm(Q - eye, 'fro') / np.linalg.norm(Q, 'fro')
    ax1.set_title(f'|Q|: {name.split(chr(10))[0]}\nd_IDFT={d_idft:.3f} d_I={d_eye:.3f}',
                  fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Q basis (前8行作为时域基底)
    ax2 = fig.add_subplot(gs[1, idx * 2 + 1])
    for row_idx in range(min(8, N)):
        row = np.abs(Q[row_idx, :])
        ax2.plot(row, linewidth=1.2, label=f'row {row_idx}')
    ax2.set_xlabel('Output index', fontsize=10)
    ax2.set_ylabel('|Q[row, :]|', fontsize=10)
    ax2.set_title(f'Q basis (rows): {name.split(chr(10))[0]}', fontsize=10)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

# ---- Row 2: Q vs IDFT vs I 对比 ----
# |Q - IDFT|
for idx, (name, _, _, _) in enumerate(scenarios):
    ax = fig.add_subplot(gs[2, idx * 2])
    Q = results[name]['Q']
    diff = np.abs(Q - idft)
    im = ax.imshow(diff, cmap='Reds', aspect='auto')
    ax.set_xlabel('Output index', fontsize=10)
    ax.set_ylabel('Input index', fontsize=10)
    ax.set_title(f'|Q − IDFT|: {name.split(chr(10))[0]}', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

# IDFT and I references
ax_idft = fig.add_subplot(gs[2, 2])
im = ax_idft.imshow(np.abs(idft), aspect='auto', cmap='hot')
ax_idft.set_title('IDFT (OFDM 标准基底)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax_idft, shrink=0.8)

ax_eye = fig.add_subplot(gs[2, 3])
im = ax_eye.imshow(np.abs(eye), aspect='auto', cmap='hot')
ax_eye.set_title('I/√N (TDM 标准基底)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax_eye, shrink=0.8)

plt.suptitle('GRU架构: CIR → Q 波形分化\n'
             'Conv1D + Complex GRU (1024 units) + Time Path (Doppler) → Q matrix',
             fontsize=14, fontweight='bold')

plt.savefig('architecture_and_Q_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: architecture_and_Q_analysis.png")

# ================================================================
# 4. CIR 统计对比
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, _, _, _) in enumerate(scenarios):
    ax = axes[idx]
    cir = results[name]['cir']
    cir_avg = np.mean(np.abs(cir), axis=1)  # mean over time snapshots
    l_tot = cir.shape[0]
    ax.stem(range(l_tot), cir_avg, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('Delay tap', fontsize=12)
    ax.set_ylabel('Mean |CIR|', fontsize=12)
    ax.set_title(f'Delay profile: {name.split(chr(10))[0]}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Annotate RMS delay spread
    delays = np.arange(l_tot)
    power = cir_avg**2
    total_pwr = np.sum(power)
    mean_d = np.sum(delays * power) / total_pwr
    rms_ds = np.sqrt(np.sum(power * (delays - mean_d)**2) / total_pwr)
    ax.axvline(x=mean_d, color='red', linestyle='--', alpha=0.5, label=f'RMS DS={rms_ds:.2f} taps')
    ax.legend(fontsize=9)

plt.suptitle('Delay Profile Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cir_delay_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cir_delay_profile.png")
