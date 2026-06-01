"""
OTFS Feature Analysis for 2D Conv Model
========================================
Analyzes whether the 2D Conv network learns OTFS-like waveform features:
1. Q matrix visualization at low vs high Doppler
2. BER vs SNR comparison
3. Time-frequency spreading analysis
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *

# Override speed for testing
import channel_tv
from utils.TDL_RandomDS import TDL_RandomDS

def build_channel(speed_min, speed_max):
    """Build a TDL channel with specific speed range."""
    return TDL_RandomDS(
        model='A',
        delay_spread_min=50e-9,
        delay_spread_max=300e-9,
        carrier_frequency=CARRIER_FREQ,
        min_speed=speed_min,
        max_speed=speed_max
    )

def test_model_at_speed(model, speed, ebno_db, num_trials=10):
    """Test model at a specific speed by replacing channel model temporarily."""
    original_channel = model._channel_model

    # Use narrow speed range around target
    test_channel = build_channel(max(0.1, speed - 0.5), speed + 0.5)
    model._channel_model = test_channel

    total_ber = 0.0
    for _ in range(num_trials):
        b, b_hat = model(200, ebno_db)
        from sionna.phy.utils import compute_ber
        total_ber += float(compute_ber(b, b_hat))

    model._channel_model = original_channel
    return total_ber / num_trials

def extract_q_matrix(model, speed, ebno_db=20.0):
    """Extract Q matrices for visualization at a specific speed."""
    original_channel = model._channel_model
    test_channel = build_channel(max(0.1, speed - 0.5), speed + 0.5)
    model._channel_model = test_channel

    # Run forward pass and capture Q
    # We need to access the Q matrix from the qQ_creator layer
    # The model's call returns (b, b_hat) in eval mode
    # We need to hook into the creator layer
    # For now, just run eval and extract from a custom pass

    # Create a hook to capture the Q matrix
    creator = model._qQ_creator_layer

    # Use training mode temporarily to not get BER output
    original_training = model.training
    model.training = True  # This makes call() return loss, not (b, b_hat)
    # But we want Q... Let's use a different approach

    # Direct call to the creator layer with CIR input
    from sionna.phy.channel import time_lag_discrete_time_channel, cir_to_time_channel
    import sionna.phy as sn

    batch_size = 4
    # Generate CIR at target speed
    a, tau = test_channel(batch_size,
                          model._rg.num_time_samples + model._l_tot - 1,
                          model._rg.bandwidth)
    h_time = cir_to_time_channel(model._rg.bandwidth, a, tau,
                                 l_min=model._l_min, l_max=model._l_max, normalize=True)

    # Extract multi-snapshot CIR (same as in model.call — dense uniform sampling)
    h_2d = h_time[:, 0, 0, 0, 0, :, :]               # (batch, num_time_steps, l_tot)
    total_time = tf.shape(h_2d)[1]
    stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
    indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
    pilots = tf.gather(h_2d, indices, axis=1)          # (batch, num_snapshots, l_tot)
    pilots = tf.transpose(pilots, [0, 2, 1])           # (batch, l_tot, num_snapshots)

    # Get Q matrix from creator
    Q, _ = creator(pilots, training=False)  # (batch, N, N) complex

    model._channel_model = original_channel
    model.training = original_training
    return Q.numpy()

def plot_q_matrix(Q, title, save_path):
    """Plot Q matrix magnitude and phase."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    Q_mag = np.abs(Q)
    Q_phase = np.angle(Q)
    Q_real = np.real(Q)
    Q_imag = np.imag(Q)

    im0 = axes[0].imshow(Q_mag, cmap='hot', aspect='auto')
    axes[0].set_title(f'{title}\nMagnitude')
    axes[0].set_xlabel('Column'); axes[0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Q_phase, cmap='RdBu', aspect='auto', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Phase')
    axes[1].set_xlabel('Column'); axes[1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[1])

    # Also show real part
    im2 = axes[2].imshow(Q_real, cmap='RdBu', aspect='auto',
                         vmax=max(abs(Q_real.max()), abs(Q_real.min())),
                         vmin=-max(abs(Q_real.max()), abs(Q_real.min())))
    axes[2].set_title('Real Part')
    axes[2].set_xlabel('Column'); axes[2].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def analyze_time_frequency_spreading(Q):
    """
    Analyze time-frequency spreading characteristics.
    OTFS spreads energy uniformly across the delay-Doppler grid.
    A flat magnitude response suggests OTFS-like spreading.
    """
    Q_mag = np.abs(Q)
    N = Q_mag.shape[0]

    # Column std (frequency-domain spreading)
    col_std = np.std(Q_mag, axis=0)  # std per column
    # Row std (time-domain spreading)
    row_std = np.std(Q_mag, axis=1)  # std per row

    # Flatness metric: lower std = more uniform spreading (more OTFS-like)
    overall_std = np.std(Q_mag)
    # Energy concentration: max/mean ratio
    energy_concentration = np.max(Q_mag) / (np.mean(Q_mag) + 1e-10)

    return {
        'overall_std': float(overall_std),
        'energy_concentration': float(energy_concentration),
        'col_std_mean': float(np.mean(col_std)),
        'row_std_mean': float(np.mean(row_std)),
        'mag_mean': float(np.mean(Q_mag)),
    }

def main():
    print("=" * 60)
    print("OTFS Feature Analysis — 2D Conv Model")
    print("=" * 60)

    from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
    import sionna.phy as sn
    sn.config.seed = SEED

    # Load best weights (final trained weights from Stage 3)
    weights_file = 'weights-qQ_Method_TV'
    print(f"\nLoading weights from: {weights_file}")
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)

    model = qQ_MODEL_TV(training=False)
    # Build model first
    b, b_hat = model(2, 20.0)
    model.set_weights(weights)
    print("Weights loaded successfully.")

    # ================================================================
    # 1. Q Matrix Analysis: Low vs High Doppler
    # ================================================================
    print("\n" + "=" * 40)
    print("1. Q Matrix at Low vs High Doppler")
    print("=" * 40)

    speeds = {'Low (3 m/s)': 3.0, 'High (120 m/s)': 120.0}
    q_matrices = {}
    metrics = {}

    for label, speed in speeds.items():
        print(f"\nExtracting Q at {label}...")
        Q_batch = extract_q_matrix(model, speed)
        # Average over batch
        Q_avg = np.mean(Q_batch, axis=0)
        q_matrices[label] = Q_avg

        # Compute metrics
        m = analyze_time_frequency_spreading(Q_avg)
        metrics[label] = m
        print(f"  Overall STD: {m['overall_std']:.4f}")
        print(f"  Energy concentration: {m['energy_concentration']:.2f}x")
        print(f"  Col STD mean: {m['col_std_mean']:.4f}")
        print(f"  Row STD mean: {m['row_std_mean']:.4f}")

        # Plot
        plot_q_matrix(Q_avg,
                      f'Q Matrix — {label}\nSTD={m["overall_std"]:.3f}',
                      f'Q_matrix_{label.replace(" ", "_").replace("/","_")}.png')

    # Compare
    print("\n--- Doppler Comparison ---")
    low = metrics['Low (3 m/s)']
    high = metrics['High (120 m/s)']
    print(f"Low speed  STD: {low['overall_std']:.4f}, Concentration: {low['energy_concentration']:.1f}x")
    print(f"High speed STD: {high['overall_std']:.4f}, Concentration: {high['energy_concentration']:.1f}x")

    if high['energy_concentration'] < low['energy_concentration']:
        print("✓ High-Doppler Q is MORE spread out (lower concentration) — OTFS-like behavior!")
    else:
        print("✗ High-Doppler Q is NOT more spread out")

    if high['overall_std'] < low['overall_std']:
        print("✓ High-Doppler Q has more uniform magnitude (lower STD) — consistent with OTFS spreading")
    else:
        print("✗ High-Doppler Q is more concentrated in magnitude")

    # ================================================================
    # 2. Q Matrix Difference Analysis
    # ================================================================
    print("\n" + "=" * 40)
    print("2. Q Matrix Difference (High - Low Doppler)")
    print("=" * 40)

    Q_diff = q_matrices['High (120 m/s)'] - q_matrices['Low (3 m/s)']
    plot_q_matrix(Q_diff,
                  'Q Difference: High(120m/s) — Low(3m/s)',
                  'Q_matrix_difference.png')
    print(f"  Q diff max magnitude: {np.max(np.abs(Q_diff)):.4f}")
    print(f"  Q diff mean magnitude: {np.mean(np.abs(Q_diff)):.4f}")
    print(f"  (Non-zero diff means network adapts Q to Doppler)")

    # ================================================================
    # 3. BER vs SNR Comparison
    # ================================================================
    print("\n" + "=" * 40)
    print("3. BER vs SNR")
    print("=" * 40)

    snr_range = np.arange(0, 26, 5)
    ber_results = {}

    for label, speed in speeds.items():
        print(f"\nTesting at {label}:")
        ber_list = []
        for snr in snr_range:
            ber = test_model_at_speed(model, speed, float(snr), num_trials=5)
            ber_list.append(ber)
            print(f"  SNR={snr:2.0f}dB  BER={ber:.4f}")
        ber_results[label] = ber_list

    # Plot BER vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ber_list in ber_results.items():
        ax.semilogy(snr_range, ber_list, 'o-', linewidth=2, markersize=8, label=label)
    ax.set_xlabel('SNR [dB]', fontsize=14)
    ax.set_ylabel('BER', fontsize=14)
    ax.set_title('BER vs SNR — 2D Conv Model', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_ylim([1e-3, 1])
    plt.tight_layout()
    plt.savefig('BER_vs_SNR.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: BER_vs_SNR.png")

    # ================================================================
    # 4. Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("OTFS Feature Analysis — Summary")
    print("=" * 60)
    print(f"""
    1. Q Matrix Spread Analysis:
       - Low Doppler (3 m/s):  STD={low['overall_std']:.3f}, Concentration={low['energy_concentration']:.1f}x
       - High Doppler (120 m/s): STD={high['overall_std']:.3f}, Concentration={high['energy_concentration']:.1f}x
       → {"Network ADAPTS Q to Doppler (OTFS-like)" if abs(low['energy_concentration']-high['energy_concentration'])>0.5 else "Network shows LIMITED Doppler adaptation"}

    2. Q Matrix Difference:
       - Mean absolute difference: {np.mean(np.abs(Q_diff)):.4f}
       → {"Significant Q variation with speed" if np.mean(np.abs(Q_diff))>0.05 else "Q is relatively static across speeds"}

    3. BER Performance:
       - Low Doppler (3 m/s):  {ber_results['Low (3 m/s)'][-1]:.4f} @ 25dB
       - High Doppler (120 m/s): {ber_results['High (120 m/s)'][-1]:.4f} @ 25dB
    """)

    print("Analysis complete. Check generated PNG files.")

if __name__ == "__main__":
    main()
