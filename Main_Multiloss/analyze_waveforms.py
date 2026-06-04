"""
4-Loss Uncertainty-Weighted Waveform Analysis.
Analyzes how the 4-loss framework (BCE+PAPR+OOB+AF) learns waveforms.

Generates summary and detailed visualization plots.
"""
import sys, os, pickle, glob, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import tensorflow as tf
import sionna.phy as sn
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.utils import compute_ber
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import SEED, CARRIER_FREQ, FFT_SIZE

sn.config.seed = SEED + 456

OUT_DIR = "waveforms_ds_scan_4loss"
os.makedirs(OUT_DIR, exist_ok=True)

DELAY_SPREADS = [10e-9, 50e-9, 100e-9, 200e-9, 300e-9, 450e-9, 620e-9, 1000e-9]
N = FFT_SIZE

# Load model
print("Loading 4-Loss model...")
model = qQ_MODEL(training=True, visulaize=False)
model(1, 40.0)
with open('weights-qQ_Method_4Loss', 'rb') as f:
    model.set_weights(pickle.load(f))
print("  Model loaded.")

# Data collection
results = {'ds_ns': [], 'ber': [], 'bce': [], 'papr': [], 'oob': [], 'af': [],
           'ls_bce': [], 'ls_papr': [], 'ls_oob': [], 'ls_af': [],
           'x_time_ccdf': {}, 'rms_ds': {}}

for ds in DELAY_SPREADS:
    ds_ns = int(ds * 1e9)
    print(f"\nProcessing {ds_ns} ns...")
    tdl_ch = TDL(model="A", delay_spread=ds, carrier_frequency=CARRIER_FREQ,
                 min_speed=0.0, max_speed=0.0)
    model._channel_model = tdl_ch

    # Training metrics
    bce_v, papr_v, oob_v, af_v = [], [], [], []
    ls_b, ls_p, ls_o, ls_a = [], [], [], []
    model.training = True
    for _ in range(10):
        total, bce, papr, oob, af, l_b, l_p, l_o, l_a = model(256, 20.0)
        bce_v.append(float(bce)); papr_v.append(float(papr))
        oob_v.append(float(oob)); af_v.append(float(af))
        ls_b.append(float(l_b)); ls_p.append(float(l_p))
        ls_o.append(float(l_o)); ls_a.append(float(l_a))

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
    model.training = True; model.CCDF_mode = True
    x_time_data, rms_ds_data = model(200, 20.0)
    model.CCDF_mode = False

    # Q matrix visualization
    model.visulaize_progress = True
    try: model(1, 40.0)
    except: pass
    model.visulaize_progress = False

    for pattern, prefix in [('_qQ_functions_Time_Visualization_*', f'Q_Time_{ds_ns}ns'),
                              ('_qQ_functions_Freq_Visualization_*', f'Q_Freq_{ds_ns}ns'),
                              ('_ofdm_model_csi_estimation_*', f'CSI_{ds_ns}ns'),
                              ('_ds_vs_*', f'{ds_ns}ns_weight')]:
        for f in glob.glob(pattern):
            os.rename(f, os.path.join(OUT_DIR, f'{prefix}.png'))

    results['ds_ns'].append(ds_ns); results['ber'].append(avg_ber)
    results['bce'].append(np.mean(bce_v)); results['papr'].append(np.mean(papr_v))
    results['oob'].append(np.mean(oob_v)); results['af'].append(np.mean(af_v))
    results['ls_bce'].append(np.mean(ls_b)); results['ls_papr'].append(np.mean(ls_p))
    results['ls_oob'].append(np.mean(ls_o)); results['ls_af'].append(np.mean(ls_a))
    results['x_time_ccdf'][ds_ns] = x_time_data.numpy()
    results['rms_ds'][ds_ns] = rms_ds_data.numpy().flatten()

    print(f"  BER={avg_ber:.5f} BCE={np.mean(bce_v):.3f} PAPR={np.mean(papr_v):.4f} "
          f"OOB={np.mean(oob_v):.3f} AF={np.mean(af_v):.4f}")
    print(f"  w_bce={np.exp(-np.mean(ls_b)):.3f} w_papr={np.exp(-np.mean(ls_p)):.1f} "
          f"w_oob={np.exp(-np.mean(ls_o)):.3f} w_af={np.exp(-np.mean(ls_a)):.1f}")

ds_arr = np.array(results['ds_ns'])

# ========== Generate Summary Plots ==========
print(f"\nGenerating plots...")

# 1: BER vs Delay Spread
fig, ax = plt.subplots(figsize=(10,6))
ax.semilogy(ds_arr, results['ber'], 'o-', lw=2.5, ms=10, color='#2196F3')
ax.set_xlabel('Delay Spread [ns]', fontsize=14); ax.set_ylabel('BER @ 20dB', fontsize=14)
ax.set_title('4-Loss Model: BER vs Delay Spread', fontsize=16); ax.grid(True, ls=':', alpha=0.7)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'01_ber_vs_ds.png'),dpi=200); plt.close()

# 2: Four Losses vs Delay Spread
fig, axes = plt.subplots(2,2,figsize=(14,10))
for ax, key, color, title in [
    (axes[0,0], 'bce', '#E53935', 'L_BCE (Communication)'),
    (axes[0,1], 'papr', '#FF9800', 'L_PAPR (Peak Power)'),
    (axes[1,0], 'oob', '#43A047', 'L_OOB (Bandwidth)'),
    (axes[1,1], 'af', '#7B1FA2', 'L_AF (Ambiguity Shape)')]:
    ax.plot(ds_arr, results[key], 'o-', lw=2.5, ms=10, color=color)
    ax.set_xlabel('Delay Spread [ns]', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14); ax.grid(True, ls=':', alpha=0.7)
fig.suptitle('4-Loss Framework: Individual Losses vs Delay Spread', fontsize=16, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'02_four_losses.png'),dpi=200); plt.close()

# 3: Uncertainty Weights vs Delay Spread
fig, axes = plt.subplots(2,2,figsize=(14,10))
w_bce  = [np.exp(-x) for x in results['ls_bce']]
w_papr = [np.exp(-x) for x in results['ls_papr']]
w_oob  = [np.exp(-x) for x in results['ls_oob']]
w_af   = [np.exp(-x) for x in results['ls_af']]
for ax, w, color, title in [
    (axes[0,0], w_bce,  '#E53935', 'w_BCE (precision)'),
    (axes[0,1], w_papr, '#FF9800', 'w_PAPR (precision)'),
    (axes[1,0], w_oob,  '#43A047', 'w_OOB (precision)'),
    (axes[1,1], w_af,   '#7B1FA2', 'w_AF (precision)')]:
    ax.plot(ds_arr, w, 'o-', lw=2.5, ms=10, color=color)
    ax.set_xlabel('Delay Spread [ns]', fontsize=12); ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(title, fontsize=14); ax.grid(True, ls=':', alpha=0.7)
fig.suptitle('Uncertainty Network: Adaptive Weights vs Delay Spread', fontsize=16, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'03_uncertainty_weights.png'),dpi=200); plt.close()

# 4: PAPR CCDF
fig, ax = plt.subplots(figsize=(12,8))
cmap = plt.get_cmap('coolwarm')
ds_list = sorted(results['x_time_ccdf'].keys())
norm = Normalize(vmin=min(ds_list), vmax=max(ds_list))
for ds_ns in ds_list:
    x_t = results['x_time_ccdf'][ds_ns]
    papr_samples = []
    for b_idx in range(min(50, x_t.shape[0])):
        x_sample = x_t[b_idx]
        p_inst = np.abs(x_sample)**2; p_mean = np.mean(p_inst)
        if p_mean > 1e-10: papr_samples.append(10*np.log10(np.max(p_inst)/p_mean))
    if papr_samples:
        s = np.sort(papr_samples)[::-1]; ccdf = np.arange(1,len(s)+1)/len(s)
        ax.semilogy(s, ccdf, color=cmap(norm(ds_ns)), lw=1.5, alpha=0.8)
ax.set_xlabel('PAPR [dB]', fontsize=14); ax.set_ylabel('CCDF', fontsize=14)
ax.set_title('PAPR CCDF Across Delay Spreads', fontsize=16)
ax.grid(True, ls=':', alpha=0.5); ax.set_ylim(1e-2,1.5)
sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=ax, label='Delay Spread [ns]')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'04_papr_ccdf.png'),dpi=200); plt.close()

# 5: OOB Spectrum Analysis for min/max DS
fig, axes = plt.subplots(1,2,figsize=(14,5))
for idx, (ds_val, label) in enumerate([(10e-9, 'Flat (10ns)'), (1000e-9, 'Dispersive (1000ns)')]):
    ds_ns = int(ds_val*1e9)
    if ds_ns in results['x_time_ccdf']:
        x_t = results['x_time_ccdf'][ds_ns][0]
        X = np.abs(np.fft.fftshift(np.fft.fft(x_t)))**2
        axes[idx].plot(X/np.max(X), color='#2196F3', lw=1.5)
        N_total = len(X); N_inband = N
        start = (N_total-N_inband)//2; end = start+N_inband
        axes[idx].axvspan(start, end, alpha=0.15, color='green', label='In-band')
        axes[idx].axvspan(0, start, alpha=0.15, color='red'); axes[idx].axvspan(end, N_total, alpha=0.15, color='red')
        axes[idx].set_title(f'{label}', fontsize=14); axes[idx].set_xlabel('Freq bin'); axes[idx].set_ylabel('Power')
        axes[idx].legend(fontsize=10); axes[idx].grid(True, ls=':', alpha=0.5)
fig.suptitle('OOB Analysis: Spectrum at Extreme Delay Spreads', fontsize=16)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'05_oob_spectrum.png'),dpi=200); plt.close()

# 6: AF Analysis (autocorrelation at min/max DS)
fig, axes = plt.subplots(1,2,figsize=(14,5))
for idx, (ds_val, label) in enumerate([(10e-9, 'Flat (10ns)'), (1000e-9, 'Dispersive (1000ns)')]):
    ds_ns = int(ds_val*1e9)
    if ds_ns in results['x_time_ccdf']:
        x_t = results['x_time_ccdf'][ds_ns][0]
        X = np.fft.fft(x_t); psd = np.abs(X)**2
        R = np.abs(np.fft.ifft(psd)); R = R / (R[0]+1e-10)
        axes[idx].stem(range(len(R)//4), R[:len(R)//4], linefmt='#7B1FA2', markerfmt='o')
        axes[idx].set_title(f'{label}', fontsize=14); axes[idx].set_xlabel('Lag τ'); axes[idx].set_ylabel('|R(τ)|')
        axes[idx].grid(True, ls=':', alpha=0.5)
fig.suptitle('AF Shape Analysis: Autocorrelation (Delay Profile)', fontsize=16)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'06_af_autocorr.png'),dpi=200); plt.close()

# 7: Summary Dashboard
fig, axes = plt.subplots(2,3,figsize=(18,12))
axes[0,0].semilogy(ds_arr, results['ber'], 'o-', color='#2196F3', lw=2, ms=8); axes[0,0].set_title('BER'); axes[0,0].grid(True,ls=':',alpha=0.5)
axes[0,1].plot(ds_arr, results['bce'], 'o-', color='#E53935', lw=2, ms=8, label='BCE')
axes[0,1].plot(ds_arr, [o*10 for o in results['oob']], 's--', color='#43A047', lw=2, ms=8, label='OOB×10')
axes[0,1].plot(ds_arr, [a*50 for a in results['af']], '^:', color='#7B1FA2', lw=2, ms=8, label='AF×50')
axes[0,1].set_title('Loss Components'); axes[0,1].legend(); axes[0,1].grid(True,ls=':',alpha=0.5)
axes[0,2].plot(ds_arr, w_bce, 'o-', color='#E53935', lw=2, ms=8, label='BCE')
axes[0,2].plot(ds_arr, w_oob, 's--', color='#43A047', lw=2, ms=8, label='OOB')
axes[0,2].plot(ds_arr, w_af, '^:', color='#7B1FA2', lw=2, ms=8, label='AF')
axes[0,2].set_title('Uncertainty Weights'); axes[0,2].legend(); axes[0,2].grid(True,ls=':',alpha=0.5)

# PAPR avg
papr_avgs = []
for ds_ns in ds_list:
    x_t = results['x_time_ccdf'][ds_ns]
    vals = [10*np.log10(np.max(np.abs(x_t[b])**2)/np.mean(np.abs(x_t[b])**2)) for b in range(min(200,x_t.shape[0])) if np.mean(np.abs(x_t[b])**2)>1e-10]
    papr_avgs.append(np.mean(vals))
axes[1,0].plot(ds_list, papr_avgs, 'o-', color='#FF5722', lw=2, ms=8); axes[1,0].set_title('Avg PAPR'); axes[1,0].grid(True,ls=':',alpha=0.5)
axes[1,1].plot(ds_arr, results['ls_bce'], 'o-', color='#E53935', lw=2, ms=8, label='BCE')
axes[1,1].plot(ds_arr, results['ls_oob'], 's--', color='#43A047', lw=2, ms=8, label='OOB')
axes[1,1].plot(ds_arr, results['ls_af'], '^:', color='#7B1FA2', lw=2, ms=8, label='AF')
axes[1,1].set_title('logσ² Values'); axes[1,1].legend(); axes[1,1].grid(True,ls=':',alpha=0.5)
axes[1,2].scatter(papr_avgs, results['ber'], c=ds_list, cmap='coolwarm', s=100, edgecolors='k')
axes[1,2].set_xlabel('PAPR [dB]'); axes[1,2].set_ylabel('BER'); axes[1,2].set_yscale('log'); axes[1,2].set_title('PAPR-BER Trade-off')
plt.colorbar(ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=Normalize(vmin=min(ds_list),vmax=max(ds_list))), ax=axes[1,2], label='DS [ns]')
fig.suptitle('4-Loss Uncertainty-Weighted Model: Analysis Dashboard', fontsize=18, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'07_summary_dashboard.png'),dpi=200); plt.close()

# Cleanup temp files
for pat in ['_qQ_*','_ofdm_*','_ds_vs_*']:
    for f in glob.glob(pat):
        try: os.remove(f)
        except: pass

# Save results
json.dump({k:v for k,v in results.items() if k not in ('x_time_ccdf','rms_ds')},
          open(os.path.join(OUT_DIR,'results.json'),'w'), indent=2, default=str)

count = len(os.listdir(OUT_DIR))
print(f"\nAnalysis complete! {count} files in {OUT_DIR}/")
print(f"  Best BER: {min(results['ber']):.5f} at {ds_arr[np.argmin(results['ber'])]} ns")
print(f"  OOB range: {min(results['oob']):.3f} - {max(results['oob']):.3f}")
print(f"  AF range:  {min(results['af']):.4f} - {max(results['af']):.4f}")
