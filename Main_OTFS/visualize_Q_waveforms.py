#!/usr/bin/env python3
"""
Q 矩阵波形可视化 — 含输入 CIR
================================
三列: 平坦 3m/s | 多径 3m/s (620ns) | 多径 120m/s (620ns)
四行: CIR 输入 | Q 时域基底 | Q 频谱 | |Q| 热力图
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

def compute_idft(N):
    k = np.arange(N); n = np.arange(N).reshape(-1, 1)
    return np.exp(1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def extract_data(model, speed, ds_min, ds_max, n_samples=64):
    from utils.TDL_RandomDS import TDL_RandomDS
    test_ch = TDL_RandomDS(model='A', delay_spread_min=ds_min, delay_spread_max=ds_max,
                           carrier_frequency=CARRIER_FREQ,
                           min_speed=max(0.1, speed-0.5), max_speed=speed+0.5)
    orig = model._channel_model
    model._channel_model = test_ch
    a, tau = test_ch(n_samples, model._rg.num_time_samples + model._l_tot - 1,
                     model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                  l_min=model._l_min, l_max=model._l_max, normalize=True)
    h_2d = h_time[:, 0, 0, 0, 0, :, :]
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    cir_snap = tf.gather(h_2d, indices, axis=1)
    cir_input = tf.transpose(cir_snap, [0, 2, 1])
    Q, _ = model._qQ_creator_layer(cir_input, training=False)

    h_first = tf.abs(cir_input[:, :, 0])
    delays = tf.cast(tf.range(tf.shape(h_first)[1]), tf.float32)
    power = tf.square(h_first)
    total_power = tf.reduce_sum(power, axis=-1, keepdims=True) + 1e-10
    mean_delay = tf.reduce_sum(delays[None, :] * power, axis=-1, keepdims=True) / total_power
    rms_ds = tf.sqrt(tf.reduce_sum(power * tf.square(delays[None, :] - mean_delay), axis=-1)
                     / tf.squeeze(total_power, -1))
    model._channel_model = orig
    return cir_input.numpy(), Q.numpy(), rms_ds.numpy()

# ================================================================
from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
model = qQ_MODEL_TV(training=False)
model(2, 40.0)
with open('weights-qQ_Method_TV', 'rb') as f:
    model.set_weights(pickle.load(f))
print("Model loaded.")

N = FFT_SIZE
IDFT = compute_idft(N)
EYE  = np.eye(N, dtype=np.complex128) / np.sqrt(N)
HOW_MANY = 8

scenarios = [
    ("flat 3m/s, DS=10ns",    3.0,  10e-9,  11e-9),
    ("multipath 3m/s, DS=620ns",  3.0,  620e-9, 621e-9),
    ("multipath 120m/s, DS=620ns", 120.0, 620e-9, 621e-9),
]

all_data = {}
for name, speed, ds_min, ds_max in scenarios:
    print(f"Extracting: {name} ...")
    cir, Q_batch, rms_ds = extract_data(model, speed, ds_min, ds_max, 64)
    idx_max = np.argmax(rms_ds)
    all_data[name] = {
        'cir': cir, 'Q': Q_batch[idx_max],
        'rms_ds': rms_ds, 'idx_max': idx_max,
    }
    d_idft = np.linalg.norm(Q_batch[idx_max] - IDFT, 'fro') / np.linalg.norm(Q_batch[idx_max], 'fro')
    d_eye  = np.linalg.norm(Q_batch[idx_max] - EYE, 'fro') / np.linalg.norm(Q_batch[idx_max], 'fro')
    print(f"  max-DS: dist_IDFT={d_idft:.3f}, dist_I={d_eye:.3f}, RMS_DS={rms_ds[idx_max]:.2f}")

# ================================================================
# Main figure: 4 rows × 3 cols
# ================================================================
fig = plt.figure(figsize=(24, 20))
gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ---- Row 0: CIR input — 2 sub-rows (heatmap + delay profile) ----
for col, (name, _, _, _) in enumerate(scenarios):
    d = all_data[name]
    cir0 = np.abs(d['cir'][d['idx_max']])  # first sample, shape (l_tot, 12)
    l_tot, n_snap = cir0.shape

    # CIR heatmap
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(cir0, aspect='auto', cmap='hot', origin='lower',
                   extent=[0, n_snap-1, 0, l_tot-1])
    ax.set_xlabel('Time snapshot', fontsize=10)
    ax.set_ylabel('Delay tap', fontsize=10)
    rms = d['rms_ds'][d['idx_max']]
    ax.set_title(f'CIR input: {name}\n{l_tot} taps × {n_snap} snapshots, RMS_DS={rms:.2f}',
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

# ---- Row 1: CIR delay profile + time evolution ----
for col, (name, _, _, _) in enumerate(scenarios):
    d = all_data[name]
    cir0 = np.abs(d['cir'][d['idx_max']])

    ax = fig.add_subplot(gs[1, col])
    # Plot each tap's time evolution
    tap_energy = np.sum(cir0, axis=1)
    top_taps = np.argsort(tap_energy)[-5:]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, tap in enumerate(top_taps):
        ax.plot(cir0[tap, :], color=colors[i], linewidth=1.2,
                label=f'tap {tap}', marker='.', markersize=3)
    ax.set_xlabel('Time snapshot', fontsize=10)
    ax.set_ylabel('|CIR|', fontsize=10)
    ax.set_title(f'CIR time evolution: {name}', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

# ---- Row 2: Q time-domain basis ----
for col, (name, _, _, _) in enumerate(scenarios):
    d = all_data[name]
    Q = d['Q']
    ax = fig.add_subplot(gs[2, col])
    for i in range(HOW_MANY):
        row = Q[i, :]
        ax.plot(np.real(row), linewidth=0.8, alpha=0.6, label=f'Re{i}')
        ax.plot(np.imag(row), linewidth=0.8, alpha=0.6, linestyle='--', label=f'Im{i}')
    d_idft = np.linalg.norm(Q - IDFT, 'fro') / np.linalg.norm(Q, 'fro')
    d_eye  = np.linalg.norm(Q - EYE, 'fro') / np.linalg.norm(Q, 'fro')
    ax.set_xlabel('n', fontsize=10); ax.set_ylabel('Q[k,n]', fontsize=10)
    ax.set_title(f'Q time-domain basis\n{name}\nd_IDFT={d_idft:.3f}  d_I={d_eye:.3f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=5, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

# ---- Row 3: Q spectrum ----
M_fft = 2048
freqs = np.fft.fftshift(np.fft.fftfreq(M_fft, d=1.0))
for col, (name, _, _, _) in enumerate(scenarios):
    d = all_data[name]
    Q = d['Q']
    ax = fig.add_subplot(gs[3, col])
    for i in range(HOW_MANY):
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(Q[i, :], n=M_fft)))
        ax.plot(freqs, spectrum, linewidth=0.8, alpha=0.6, label=f'Row{i}')
    ax.set_xlabel('Normalized freq', fontsize=10); ax.set_ylabel('|Spectrum|', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title(f'Q spectrum (FFT of rows)\n{name}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=5, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('CIR Input → Q Waveform | GRU Architecture | DS: 10ns / 620ns / 620ns',
             fontsize=14, fontweight='bold')
plt.savefig('Q_waveform_inspection.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: Q_waveform_inspection.png")

# ================================================================
# Separate CIR detail figure
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for col, (name, _, _, _) in enumerate(scenarios):
    d = all_data[name]
    cir0 = np.abs(d['cir'][d['idx_max']])
    # Average over time → delay profile
    cir_avg = np.mean(cir0, axis=1)
    ax = axes[col]
    ax.stem(range(len(cir_avg)), cir_avg, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('Delay tap', fontsize=11)
    ax.set_ylabel('Mean |CIR|', fontsize=11)
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle('CIR Delay Profile (averaged over 12 snapshots)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cir_input_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cir_input_comparison.png")
