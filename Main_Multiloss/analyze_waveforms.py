"""
Comprehensive analysis of uncertainty-guided model.
Analyzes how the network learns different waveforms across channel conditions.

Generates:
  1. Q matrix time-domain waveforms at different delay spreads
  2. Q matrix frequency-domain spectra
  3. Channel frequency responses
  4. PAPR CCDF curves across delay spreads
  5. lambda_papr (uncertainty weight) vs delay spread
  6. BER vs delay spread
  7. Summary dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import pickle
import tensorflow as tf
import sionna.phy as sn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.utils import compute_ber
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import SEED, CARRIER_FREQ, FFT_SIZE

sn.config.seed = SEED + 123

# ========== Output directory ==========
OUT_DIR = "waveforms_ds_scan_uncertainty"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== Parameters ==========
DELAY_SPREADS = [10e-9, 50e-9, 100e-9, 200e-9, 300e-9, 450e-9, 620e-9, 1000e-9]
N = FFT_SIZE
HOW_MANY_WAVES = 8
M_FFT = 2048 * N

# ========== Load model ==========
print("Loading uncertainty-guided model...")
model = qQ_MODEL(training=True, visulaize=False)
model(1, 40.0)
with open('weights-qQ_Method_BCE_Uncertainty', 'rb') as f:
    model.set_weights(pickle.load(f))
print("  Model loaded.")

# Also load complexity model if available for comparison
# Note: architecture differs (no uncertainty layers), so we can't load directly.
# Use separate evaluation instead.
has_complexity = False  # Architecture mismatch prevents direct comparison
model_cpx = None

# ========== Data collection ==========
results = {
    'ds_ns': [], 'ber': [], 'bce': [], 'papr': [], 'lambda_papr': [],
    'q_waveforms': {}, 'h_freq': {}, 'x_time_ccdf': {}, 'rms_ds': {}
}

# For complexity comparison (disabled due to architecture mismatch)
ber_cpx = None

for ds in DELAY_SPREADS:
    ds_ns = int(ds * 1e9)
    print(f"\nProcessing delay spread: {ds_ns} ns...")

    tdl_ch = TDL(model="A", delay_spread=ds, carrier_frequency=CARRIER_FREQ,
                 min_speed=0.0, max_speed=0.0)
    model._channel_model = tdl_ch

    # Collect training metrics (multiple runs for stability)
    bce_vals, papr_vals, lam_vals = [], [], []
    for _ in range(10):
        model.training = True
        total, bce, papr, lam = model(256, 20.0)
        bce_vals.append(float(bce))
        papr_vals.append(float(papr))
        lam_vals.append(float(lam))

    # BER evaluation
    model_eval = qQ_MODEL(training=False)
    model_eval(300, 20.0)
    model_eval.set_weights(model.get_weights())
    model_eval._channel_model = tdl_ch
    total_ber = 0.0
    for _ in range(5):
        b, b_hat = model_eval(300, 20.0)
        total_ber += float(compute_ber(b, b_hat))
    avg_ber = total_ber / 5

    # CCDF data
    model.training = True
    model.CCDF_mode = True
    x_time_data, rms_ds_data = model(200, 20.0)
    model.CCDF_mode = False

    # Get Q matrix via visualization mode (single sample)
    model.training = True
    model.visulaize_progress = True
    try:
        model(1, 40.0)
    except:
        pass
    model.visulaize_progress = False

    # Rename and move generated visualization files
    import glob
    time_files = glob.glob('_qQ_functions_Time_Visualization_*.png')
    freq_files = glob.glob('_qQ_functions_Freq_Visualization_*.png')
    csi_files = glob.glob('_ofdm_model_csi_estimation_*.png')
    weight_files = glob.glob('_ds_vs_*.png')

    for f in time_files:
        os.rename(f, os.path.join(OUT_DIR, f'Q_Time_{ds_ns}ns.png'))
    for f in freq_files:
        os.rename(f, os.path.join(OUT_DIR, f'Q_Freq_{ds_ns}ns.png'))
    for f in csi_files:
        os.rename(f, os.path.join(OUT_DIR, f'CSI_{ds_ns}ns.png'))
    for f in weight_files:
        os.rename(f, os.path.join(OUT_DIR, f'{ds_ns}ns_{f}'))

    # Store results
    results['ds_ns'].append(ds_ns)
    results['ber'].append(avg_ber)
    results['bce'].append(np.mean(bce_vals))
    results['papr'].append(np.mean(papr_vals))
    results['lambda_papr'].append(np.mean(lam_vals))
    results['x_time_ccdf'][ds_ns] = x_time_data.numpy()
    results['rms_ds'][ds_ns] = rms_ds_data.numpy().flatten()

    print(f"  BER@20dB: {avg_ber:.5f}, BCE: {np.mean(bce_vals):.4f}, "
          f"PAPR: {np.mean(papr_vals):.4f}, λ: {np.mean(lam_vals):.5f}")

# ========== Generate Summary Plots ==========
print(f"\n{'='*60}")
print("Generating analysis plots...")
print(f"{'='*60}")

ds_arr = np.array(results['ds_ns'])

# --- Plot 1: BER vs Delay Spread ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(ds_arr, results['ber'], 'o-', linewidth=2.5, markersize=10,
            color='#2196F3', label='Uncertainty-Guided Model')
ax.set_xlabel('Delay Spread [ns]', fontsize=14)
ax.set_ylabel('BER @ 20dB', fontsize=14)
ax.set_title('BER vs Delay Spread', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_ber_vs_delay_spread.png'), dpi=200)
plt.close()
print("  [1/10] BER vs Delay Spread")

# --- Plot 2: Loss Components vs Delay Spread ---
fig, ax1 = plt.subplots(figsize=(10, 6))
color1, color2 = '#E53935', '#43A047'
ax1.set_xlabel('Delay Spread [ns]', fontsize=14)
ax1.set_ylabel('BCE Loss', color=color1, fontsize=14)
ax1.plot(ds_arr, results['bce'], 'o-', color=color1, linewidth=2, markersize=10)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
ax2.set_ylabel('PAPR Loss (×100)', color=color2, fontsize=14)
ax2.plot(ds_arr, [p*100 for p in results['papr']], 's--', color=color2,
         linewidth=2, markersize=10)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('BCE & PAPR Loss vs Delay Spread', fontsize=16)
ax1.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_loss_vs_ds.png'), dpi=200)
plt.close()
print("  [2/10] Loss Components vs Delay Spread")

# --- Plot 3: lambda_papr (Uncertainty Weight) vs Delay Spread ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ds_arr, results['lambda_papr'], 'o-', color='#7B1FA2',
        linewidth=2.5, markersize=12)
ax.set_xlabel('Delay Spread [ns]', fontsize=14)
ax.set_ylabel(r'$\lambda_{PAPR}$ (Uncertainty Network Output)', fontsize=14)
ax.set_title('Adaptive PAPR Weight vs Delay Spread', fontsize=16)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_lambda_papr_vs_ds.png'), dpi=200)
plt.close()
print("  [3/10] lambda_papr vs Delay Spread")

# --- Plot 4: PAPR CCDF curves ---
fig, ax = plt.subplots(figsize=(12, 8))
cmap = plt.get_cmap('coolwarm')
ds_list = sorted(results['x_time_ccdf'].keys())
norm = Normalize(vmin=min(ds_list), vmax=max(ds_list))

for ds_ns in ds_list:
    x_t = results['x_time_ccdf'][ds_ns]
    papr_samples = []
    for b_idx in range(min(50, x_t.shape[0])):
        x_sample = x_t[b_idx]
        p_inst = np.abs(x_sample)**2
        p_mean = np.mean(p_inst)
        if p_mean > 1e-10:
            papr_db = 10 * np.log10(np.max(p_inst) / p_mean)
            papr_samples.append(papr_db)
    if papr_samples:
        papr_sorted = np.sort(papr_samples)[::-1]
        ccdf = np.arange(1, len(papr_sorted)+1) / len(papr_sorted)
        color = cmap(norm(ds_ns))
        ax.semilogy(papr_sorted, ccdf, color=color, linewidth=1.5, alpha=0.8,
                    label=f'{ds_ns} ns')

ax.set_xlabel('PAPR [dB]', fontsize=14)
ax.set_ylabel('CCDF', fontsize=14)
ax.set_title('PAPR CCDF Across Delay Spreads', fontsize=16)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_ylim(1e-2, 1.5)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Delay Spread [ns]', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_papr_ccdf.png'), dpi=200)
plt.close()
print("  [4/10] PAPR CCDF Curves")

# --- Plot 5: Average PAPR vs Delay Spread ---
papr_means = []
for ds_ns in ds_list:
    x_t = results['x_time_ccdf'][ds_ns]
    papr_vals = []
    for b_idx in range(min(200, x_t.shape[0])):
        x_sample = x_t[b_idx]
        p_inst = np.abs(x_sample)**2
        p_mean = np.mean(p_inst)
        if p_mean > 1e-10:
            papr_vals.append(10 * np.log10(np.max(p_inst) / p_mean))
    papr_means.append(np.mean(papr_vals))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ds_list, papr_means, 'o-', color='#FF5722', linewidth=2.5, markersize=10)
ax.set_xlabel('Delay Spread [ns]', fontsize=14)
ax.set_ylabel('Average PAPR [dB]', fontsize=14)
ax.set_title('Average PAPR vs Delay Spread', fontsize=16)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_avg_papr_vs_ds.png'), dpi=200)
plt.close()
print("  [5/10] Average PAPR vs Delay Spread")

# --- Plot 6: Q Matrix Complexity Proxy ---
# Use PAPR as a proxy for waveform complexity
# Higher PAPR suggests more OFDM-like (energy spread across time)
# Lower PAPR suggests more TDM-like (energy concentrated in time)
fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()
ax.plot(ds_list, papr_means, 'o-', color='#FF5722', linewidth=2.5, markersize=10,
        label='Avg PAPR [dB]')
ax2.semilogy(ds_arr, results['ber'], 's--', color='#2196F3', linewidth=2,
             markersize=10, label='BER')
ax.set_xlabel('Delay Spread [ns]', fontsize=14)
ax.set_ylabel('Average PAPR [dB]', color='#FF5722', fontsize=14)
ax2.set_ylabel('BER @ 20dB', color='#2196F3', fontsize=14)
ax.tick_params(axis='y', labelcolor='#FF5722')
ax2.tick_params(axis='y', labelcolor='#2196F3')
fig.suptitle('Waveform Efficiency (PAPR) vs Communication Quality (BER)', fontsize=16)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='upper left')
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_papr_vs_ber_tradeoff.png'), dpi=200)
plt.close()
print("  [6/10] PAPR vs BER Trade-off")

# --- Plot 7: Waveform Region Analysis ---
# Categorize delay spreads into three regimes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
regime_labels = ['Low DS (10-50ns)\n"TDM-like"', 'Mid DS (100-300ns)\n"Hybrid"', 'High DS (450-1000ns)\n"OFDM-like"']
regime_ds = [[10, 50], [100, 200, 300], [450, 620, 1000]]
colors_regime = ['#4CAF50', '#FF9800', '#F44336']

for idx, (r_ds, label, color) in enumerate(zip(regime_ds, regime_labels, colors_regime)):
    ax = axes[idx]
    for ds_ns in r_ds:
        if ds_ns in results['x_time_ccdf']:
            x_t = results['x_time_ccdf'][ds_ns]
            x_sample = x_t[0]
            p_inst = np.abs(x_sample)**2
            p_norm = p_inst / np.mean(p_inst)
            ax.plot(p_norm[:80], linewidth=1, alpha=0.7,
                    label=f'{ds_ns} ns' if ds_ns == r_ds[0] else None)
    ax.set_title(label, fontsize=14)
    ax.set_xlabel('Time Sample Index')
    ax.set_ylabel('Normalized Instantaneous Power')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_facecolor('#FAFAFA')

fig.suptitle('Time-Domain Power Envelope Across Channel Regimes', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_waveform_regimes.png'), dpi=200)
plt.close()
print("  [7/10] Waveform Regime Analysis")

# --- Plot 8: Summary Dashboard ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1: BER
ax = axes[0, 0]
ax.semilogy(ds_arr, results['ber'], 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.set_xlabel('Delay Spread [ns]'); ax.set_ylabel('BER @ 20dB')
ax.set_title('Communication Quality'); ax.grid(True, linestyle=':', alpha=0.5)

# 2: BCE & PAPR
ax = axes[0, 1]
ax.plot(ds_arr, results['bce'], 'o-', color='#E53935', linewidth=2, markersize=8, label='BCE')
ax.plot(ds_arr, [p*100 for p in results['papr']], 's--', color='#43A047',
        linewidth=2, markersize=8, label='PAPR×100')
ax.set_xlabel('Delay Spread [ns]'); ax.set_ylabel('Loss'); ax.legend()
ax.set_title('Loss Components'); ax.grid(True, linestyle=':', alpha=0.5)

# 3: PAPR vs DS
ax = axes[0, 2]
ax.plot(ds_list, papr_means, 'o-', color='#FF5722', linewidth=2, markersize=8)
ax.set_xlabel('Delay Spread [ns]'); ax.set_ylabel('Avg PAPR [dB]')
ax.set_title('Waveform PAPR'); ax.grid(True, linestyle=':', alpha=0.5)

# 4: lambda_papr
ax = axes[1, 0]
ax.plot(ds_arr, results['lambda_papr'], 'o-', color='#7B1FA2', linewidth=2, markersize=8)
ax.set_xlabel('Delay Spread [ns]'); ax.set_ylabel(r'$\lambda_{PAPR}$')
ax.set_title('Uncertainty Network Output'); ax.grid(True, linestyle=':', alpha=0.5)

# 5: PAPR-BER tradeoff
ax = axes[1, 1]
ax.plot(papr_means, results['ber'], 'o-', color='#E91E63', linewidth=2, markersize=8)
ax.set_xlabel('Avg PAPR [dB]'); ax.set_ylabel('BER @ 20dB')
ax.set_title('PAPR-BER Relationship'); ax.set_yscale('log')
ax.grid(True, linestyle=':', alpha=0.5)

# 6: Trade-off scatter
ax = axes[1, 2]
sc = ax.scatter(papr_means, results['ber'], c=ds_list, cmap='coolwarm',
                s=100, edgecolors='k')
ax.set_xlabel('Avg PAPR [dB]'); ax.set_ylabel('BER @ 20dB')
ax.set_title('PAPR-BER Trade-off'); ax.set_yscale('log')
ax.grid(True, linestyle=':', alpha=0.5)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label('DS [ns]')

fig.suptitle('Uncertainty-Guided Multi-Task Learning: Analysis Dashboard',
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '08_summary_dashboard.png'), dpi=200)
plt.close()
print("  [8/10] Summary Dashboard")

# --- Plot 9: Detailed Q matrix analysis for 3 key delay spreads ---
for ds_val, label in [(10e-9, 'flat'), (300e-9, 'mid'), (1000e-9, 'dispersive')]:
    ds_ns = int(ds_val * 1e9)
    tdl_ch = TDL(model="A", delay_spread=ds_val, carrier_frequency=CARRIER_FREQ,
                 min_speed=0.0, max_speed=0.0)
    model._channel_model = tdl_ch
    model.training = True
    model.visulaize_progress = True
    try:
        model(1, 40.0)
    except:
        pass
    model.visulaize_progress = False

    # Move generated files
    import glob
    for pattern, prefix in [('_qQ_functions_Time_Visualization_*.png', f'Q_Time_{label}'),
                              ('_qQ_functions_Freq_Visualization_*.png', f'Q_Freq_{label}'),
                              ('_ofdm_model_csi_estimation_*.png', f'CSI_{label}')]:
        for f in glob.glob(pattern):
            os.rename(f, os.path.join(OUT_DIR, f'{prefix}_{ds_ns}ns.png'))
print("  [9/10] Detailed Q Matrix Visualizations")

# --- Plot 10: PAPR at 1% CCDF vs Delay Spread ---
papr_1pct = []
for ds_ns in ds_list:
    x_t = results['x_time_ccdf'][ds_ns]
    papr_vals = []
    for b_idx in range(min(200, x_t.shape[0])):
        x_sample = x_t[b_idx]
        p_inst = np.abs(x_sample)**2
        p_mean = np.mean(p_inst)
        if p_mean > 1e-10:
            papr_vals.append(10 * np.log10(np.max(p_inst) / p_mean))
    papr_sorted = np.sort(papr_vals)[::-1]
    n = len(papr_sorted)
    idx_1pct = max(0, int(n * 0.01))
    papr_1pct.append(papr_sorted[idx_1pct] if idx_1pct < n else papr_sorted[-1])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ds_list, papr_1pct, 'o-', color='#E91E63', linewidth=2.5, markersize=10)
ax.set_xlabel('Delay Spread [ns]', fontsize=14)
ax.set_ylabel('PAPR @ 1% CCDF [dB]', fontsize=14)
ax.set_title('PAPR (1% CCDF) vs Delay Spread', fontsize=16)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '09_papr_1pct_vs_ds.png'), dpi=200)
plt.close()
print("  [10/10] PAPR @ 1% CCDF")

# Clean up any remaining temp files
import glob
for pat in ['_qQ_*', '_ofdm_*', '_ds_vs_*']:
    for f in glob.glob(pat):
        try:
            os.remove(f)
        except:
            pass

# Save numerical results
import json
json.dump({k: v for k, v in results.items() if k not in ('x_time_ccdf', 'rms_ds', 'q_waveforms', 'h_freq')},
          open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=2, default=str)

count = len(os.listdir(OUT_DIR))
print(f"\n{'='*60}")
print(f"Analysis complete! {count} files in {OUT_DIR}/")
print(f"Key findings:")
print(f"  Best BER: {min(results['ber']):.5f} at {ds_arr[np.argmin(results['ber'])]} ns")
print(f"  PAPR range: {min(papr_means):.1f} - {max(papr_means):.1f} dB")
print(f"  lambda_papr range: {min(results['lambda_papr']):.5f} - {max(results['lambda_papr']):.5f}")
print(f"{'='*60}")
