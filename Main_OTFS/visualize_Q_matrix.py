"""
Visualize the learned Q matrix to verify OTFS-like features.
Compares Q across different speed (Doppler) conditions:
  - Low speed (~3 m/s) → should resemble IDFT (OFDM-like)
  - High speed (~120 m/s) → should show time-frequency spreading (OTFS-like)

Visualizations:
  1. Q magnitude heatmap (low vs high speed vs IDFT)
  2. Q phase heatmap
  3. Row waveform magnitudes (time-domain basis functions)
  4. Column waveform magnitudes (frequency-domain basis)
  5. Q @ Q^H deviation from identity (orthogonality check)
  6. Energy spread per row (Gini coefficient / entropy)
"""
import sys
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import tensorflow as tf
from config import *

# ---------------------------------------------------------------------------
# Helper: IDFT matrix (same as used in training)
# ---------------------------------------------------------------------------
def idft_matrix(N):
    pi = tf.constant(np.pi, dtype=tf.float32)
    zero = tf.constant(0.0, dtype=tf.float32)
    n = tf.cast(tf.range(N), tf.float32)
    k = tf.reshape(n, (-1, 1))
    phase = (2.0 * pi / tf.cast(N, tf.float32)) * n * k
    W = tf.exp(tf.complex(zero, phase))
    return W / tf.cast(tf.sqrt(tf.cast(N, tf.float32)), W.dtype)


# ---------------------------------------------------------------------------
# Load Q-creator model (the sub-network that outputs Q, not full pipeline)
# ---------------------------------------------------------------------------
def build_q_creator(weight_path=None):
    """Build the qQ_creator_conv_gru layer and optionally load weights."""
    from src.qQ_Method.qQ_creator_layer import qQ_creator_conv_gru
    from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV

    # Build full model to get weight structure
    model = qQ_MODEL_TV(training=False)
    # Dummy forward pass to build layers
    model(2, 40.0)

    if weight_path and os.path.exists(weight_path):
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
        model.set_weights(weights)
        print(f"Loaded weights from {weight_path}")
        return model._qQ_creator_layer, model
    else:
        if weight_path:
            print(f"WARNING: weight file not found at {weight_path}, showing UNTRAINED Q")
        else:
            print("No weight file provided, showing UNTRAINED Q")
        return model._qQ_creator_layer, model


# ---------------------------------------------------------------------------
# Generate CIR via direct h_time sampling (matching updated qQ_Model_TV.py)
# ---------------------------------------------------------------------------
def make_cir_input(batch_size, l_tot, speed, rg, ofdm_modulator,
                   channel_time, seed=None):
    """
    Use the actual Sionna TDL_RandomDS channel to generate multi-snapshot CIR.

    Mimics the updated qQ_Model_TV.py:
      1. Generate time-varying channel h_time via cir_to_time_channel
      2. Dense uniform sampling across time axis (NUM_TIME_SNAPSHOTS snapshots)

    Args:
        speed: float, UE speed in m/s
    Returns: complex tensor of shape [batch_size, l_tot, NUM_TIME_SNAPSHOTS]
    """
    from config import CARRIER_FREQ, DELAY_SPREAD
    from utils.TDL_RandomDS import TDL_RandomDS
    from sionna.phy.channel import cir_to_time_channel

    l_max = l_tot - 1  # l_min=0 → l_tot = l_max + 1

    tdl = TDL_RandomDS(
        model="A",
        delay_spread_min=50e-9,
        delay_spread_max=300e-9,
        carrier_frequency=CARRIER_FREQ,
        min_speed=speed,
        max_speed=speed,
    )

    num_time_steps = rg.num_time_samples + l_tot - 1
    a, tau = tdl(batch_size, num_time_steps, rg.bandwidth)
    h_time = cir_to_time_channel(rg.bandwidth, a, tau,
                                 l_min=0, l_max=l_max, normalize=True)
    # h_time: [B, 1, 1, 1, 1, num_time_steps, l_tot]

    # Sample CIR uniformly across time (same as updated model with NUM_TIME_SNAPSHOTS)
    h_2d = h_time[:, 0, 0, 0, 0, :, :]               # [B, num_time_steps, l_tot]
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    pilots_post_channel = tf.gather(h_2d, indices, axis=1)  # [B, num_snapshots, l_tot]
    pilots_post_channel = tf.transpose(pilots_post_channel, [0, 2, 1])  # [B, l_tot, num_snapshots]
    return pilots_post_channel


# ---------------------------------------------------------------------------
# Extract Q for different speed regimes
# ---------------------------------------------------------------------------
def get_Q_for_speeds(q_creator, model, l_tot,
                      speeds=None, batch_size=4):
    """Run the qQ creator layer using REAL Sionna channel CIR at different speeds."""
    if speeds is None:
        speeds = [3.0, 60.0, 120.0]
    results = {}
    for speed in speeds:
        cir = make_cir_input(
            batch_size, l_tot, speed,
            rg=model._rg,
            ofdm_modulator=model.OFDM_modulator,
            channel_time=model._channel_time
        )
        Q, _ = q_creator(cir, training=False)
        results[speed] = Q.numpy()
    return results


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def mag_db(x):
    """Convert complex matrix to dB magnitude."""
    return 20 * np.log10(np.abs(x) + 1e-10)


def plot_heatmap(ax, data, title, cmap='viridis', vrange=None):
    """Plot a 2D heatmap on given axes."""
    if vrange is None:
        vmin, vmax = np.min(data), np.max(data)
    else:
        vmin, vmax = vrange
    im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap,
                   norm=Normalize(vmin=vmin, vmax=vmax))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Column index')
    ax.set_ylabel('Row index')
    plt.colorbar(im, ax=ax)


def plot_row_waveforms(ax, Q, title, max_rows=8):
    """Plot magnitude of selected rows of Q as waveforms."""
    N = Q.shape[0]
    step = max(1, N // max_rows)
    for i in range(0, N, step):
        waveform = np.abs(Q[i, :])
        ax.plot(waveform, label=f'row {i}', alpha=0.7, linewidth=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Column index')
    ax.set_ylabel('Magnitude')
    if N <= 16:
        ax.legend(fontsize=6, ncol=2)


def plot_column_waveforms(ax, Q, title, max_cols=8):
    """Plot magnitude of selected columns of Q as waveforms."""
    N = Q.shape[0]
    step = max(1, N // max_cols)
    for i in range(0, N, step):
        waveform = np.abs(Q[:, i])
        ax.plot(waveform, label=f'col {i}', alpha=0.7, linewidth=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Row index')
    ax.set_ylabel('Magnitude')
    if N <= 16:
        ax.legend(fontsize=6, ncol=2)


def row_energy_spread(Q):
    """Compute normalized energy per element for each row (how spread out energy is)."""
    magnitudes = np.abs(Q) ** 2  # |Q[i,j]|^2
    row_energies = np.sum(magnitudes, axis=1, keepdims=True) + 1e-10
    return magnitudes / row_energies


def compute_gini_per_row(Q):
    """
    Gini coefficient per row: measures energy concentration (0 = uniform, 1 = single peak).
    Lower Gini → more spreading (OTFS-like).
    """
    N = Q.shape[0]
    ginis = []
    for i in range(N):
        power = np.sort(np.abs(Q[i, :]) ** 2)
        n = len(power)
        # Don't use the first index; follow standard definition
        gini_idx = np.arange(1, n + 1)
        gini = (2 * np.sum(gini_idx * power)) / (n * np.sum(power) + 1e-10) - (n + 1) / n
        ginis.append(gini)
    return np.array(ginis)


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------
def main():
    N = FFT_SIZE  # 32
    num_snapshots = NUM_TIME_SNAPSHOTS

    weight_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights-qQ_Method_TV'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights-qQ_Method_TV_initial'),
    ]
    weight_path = None
    for wf in weight_candidates:
        if os.path.exists(wf):
            weight_path = wf
            break

    print(f"N={N}, num_snapshots={num_snapshots}")
    q_creator, model = build_q_creator(weight_path)
    l_tot = int(model._l_tot)
    print(f"l_tot (from model) = {l_tot}")

    # Q matrices at 3 speeds using REAL Sionna channel (batch=8 → avg)
    speeds = [3.0, 60.0, 120.0]
    batch_size = 8
    print(f"Generating CIR with real Sionna channel at speeds {speeds} m/s ...")
    results = get_Q_for_speeds(q_creator, model, l_tot,
                                speeds=speeds, batch_size=batch_size)

    def col_normalize(Q):
        norms = np.sqrt(np.sum(np.abs(Q) ** 2, axis=0, keepdims=True) + 1e-10)
        return Q / norms * np.sqrt(N)

    # Average across batch then normalize
    Q_lo  = col_normalize(np.mean(results[3.0],  axis=0))
    Q_mid = col_normalize(np.mean(results[60.0], axis=0))
    Q_hi  = col_normalize(np.mean(results[120.0], axis=0))
    Q_idft_n = col_normalize(idft_matrix(N).numpy())

    # Metrics
    gini_idft = compute_gini_per_row(Q_idft_n)
    gini_lo    = compute_gini_per_row(Q_lo)
    gini_mid   = compute_gini_per_row(Q_mid)
    gini_hi    = compute_gini_per_row(Q_hi)

    print(f"\nGini coefficient (mean ± std, lower = more OTFS-like spreading):")
    print(f"  IDFT:       {np.mean(gini_idft):.4f} ± {np.std(gini_idft):.4f}")
    print(f"  3 m/s:      {np.mean(gini_lo):.4f} ± {np.std(gini_lo):.4f}")
    print(f"  60 m/s:     {np.mean(gini_mid):.4f} ± {np.std(gini_mid):.4f}")
    print(f"  120 m/s:    {np.mean(gini_hi):.4f} ± {np.std(gini_hi):.4f}")

    def ortho_deviation(Q):
        QQH = Q @ Q.conj().T
        target = np.eye(N) * np.trace(QQH).real / N
        return np.mean(np.abs(QQH - target) ** 2)

    print(f"\nOrthogonality deviation (Q @ Q^H vs scaled identity):")
    print(f"  IDFT:       {ortho_deviation(Q_idft_n):.6f}")
    print(f"  3 m/s:      {ortho_deviation(Q_lo):.6f}")
    print(f"  60 m/s:     {ortho_deviation(Q_mid):.6f}")
    print(f"  120 m/s:    {ortho_deviation(Q_hi):.6f}")

    # ===================================================================
    # Figure 1: magnitude / phase / waveforms / Gini
    # ===================================================================
    fig = plt.figure(figsize=(24, 18))
    all_mags = [mag_db(Q_idft_n), mag_db(Q_lo), mag_db(Q_mid), mag_db(Q_hi)]
    vmin = min(m.min() for m in all_mags)
    vmax = max(m.max() for m in all_mags)

    # Row 1 Magnitude
    for idx, (Q, label) in enumerate([
        (Q_idft_n, 'IDFT'), (Q_lo, 'v=3 m/s'), (Q_mid, 'v=60 m/s'), (Q_hi, 'v=120 m/s')
    ]):
        ax = fig.add_subplot(3, 4, idx + 1)
        plot_heatmap(ax, mag_db(Q), f'{label}: |Q| [dB]', vrange=(vmin, vmax))

    # Row 2 Phase
    for idx, (Q, label) in enumerate([
        (Q_idft_n, 'IDFT'), (Q_lo, 'v=3 m/s'), (Q_mid, 'v=60 m/s'), (Q_hi, 'v=120 m/s')
    ]):
        ax = fig.add_subplot(3, 4, 5 + idx)
        plot_heatmap(ax, np.angle(Q), f'{label}: Phase [rad]', cmap='twilight',
                     vrange=(-np.pi, np.pi))

    # Row 3 Row waveforms + Gini
    ax9  = fig.add_subplot(3, 4, 9)
    plot_row_waveforms(ax9, Q_idft_n, 'IDFT row |Q[i,:]|')
    ax10 = fig.add_subplot(3, 4, 10)
    plot_row_waveforms(ax10, Q_lo, 'v=3 m/s row |Q[i,:]|')
    ax11 = fig.add_subplot(3, 4, 11)
    plot_row_waveforms(ax11, Q_hi, 'v=120 m/s row |Q[i,:]|')

    ax12 = fig.add_subplot(3, 4, 12)
    rows = np.arange(N)
    w = 0.15
    ax12.bar(rows - 1.5*w, gini_idft, w, label='IDFT', alpha=0.8)
    ax12.bar(rows - 0.5*w, gini_lo,    w, label='3 m/s', alpha=0.8)
    ax12.bar(rows + 0.5*w, gini_mid,   w, label='60 m/s', alpha=0.8)
    ax12.bar(rows + 1.5*w, gini_hi,    w, label='120 m/s', alpha=0.8)
    ax12.set_title('Gini per row (↓ = more OTFS spread)', fontsize=10)
    ax12.set_xlabel('Row index'); ax12.set_ylabel('Gini')
    ax12.legend(fontsize=6); ax12.set_ylim(0, 1)

    plt.suptitle(f'Q Matrix: OTFS Verification (real Sionna channel, N={N})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    p1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q_matrix_visualization.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight'); print(f"Saved: {p1}")

    # ===================================================================
    # Figure 2: energy spread + orthogonality
    # ===================================================================
    fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
    for idx, (Q, label) in enumerate([
        (Q_idft_n, 'IDFT'), (Q_lo, '3 m/s'), (Q_mid, '60 m/s'), (Q_hi, '120 m/s')
    ]):
        spread = row_energy_spread(Q)
        im = axes2[0, idx].imshow(spread, aspect='auto', origin='lower', cmap='hot',
                                   norm=Normalize(vmin=0, vmax=np.percentile(spread, 95)))
        axes2[0, idx].set_title(f'{label}: row energy spread')
        plt.colorbar(im, ax=axes2[0, idx])

        QQH = np.abs(Q @ Q.conj().T)
        im2 = axes2[1, idx].imshow(QQH, aspect='auto', origin='lower', cmap='viridis')
        axes2[1, idx].set_title(f'{label}: |Q Q^H|')
        plt.colorbar(im2, ax=axes2[1, idx])

    plt.suptitle('Energy Spread & Orthogonality', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q_matrix_spread_ortho.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight'); print(f"Saved: {p2}")

    # ===================================================================
    # Figure 3: time-domain waveform + PAPR
    # ===================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 8))

    def make_time_waveform(Q):
        test_in = np.zeros((1, 1, 1, NUM_OFDM_SYMBOL, N), dtype=np.complex64)
        for k in [0, 4, 8, 16]:
            test_in[..., 0, k] = 1.0 + 0.0j
        return np.squeeze(np.einsum('bxyzi,ij->bxyzj', test_in, Q))

    def papr_db(x):
        p = np.abs(x)**2; return 10*np.log10(np.max(p)/(np.mean(p)+1e-10))

    x_idft = make_time_waveform(Q_idft_n)
    x_lo   = make_time_waveform(Q_lo)
    x_hi   = make_time_waveform(Q_hi)

    for ax, x, label, c in [
        (axes3[0,0], x_idft, 'IDFT', 'blue'),
        (axes3[0,1], x_lo,   '3 m/s', 'green'),
        (axes3[1,0], x_hi,   '120 m/s', 'red'),
    ]:
        ax.plot(np.abs(x), color=c, alpha=0.8, linewidth=0.8)
        ax.set_title(f'{label}: time-domain |waveform|')

    cats  = ['IDFT', '3 m/s', '60 m/s', '120 m/s']
    paprs = [papr_db(make_time_waveform(Q_idft_n)),
             papr_db(make_time_waveform(Q_lo)),
             papr_db(make_time_waveform(Q_mid)),
             papr_db(make_time_waveform(Q_hi))]
    cols  = ['blue', 'green', 'orange', 'red']
    axes3[1,1].bar(cats, paprs, color=cols, alpha=0.7)
    axes3[1,1].set_title('PAPR'); axes3[1,1].set_ylabel('dB')
    for i,v in enumerate(paprs):
        axes3[1,1].text(i, v+0.1, f'{v:.1f}', ha='center', fontsize=9)

    print(f"\nPAPR: IDFT={paprs[0]:.1f}  3m/s={paprs[1]:.1f}  60m/s={paprs[2]:.1f}  120m/s={paprs[3]:.1f} dB")
    plt.suptitle('Time-Domain Waveform & PAPR', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    p3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q_matrix_waveform.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight'); print(f"Saved: {p3}")


if __name__ == '__main__':
    main()
