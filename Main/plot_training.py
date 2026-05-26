"""
Plot training metrics from Stage 1/2/3 CSV files.
Usage: python plot_training.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, 'training_curves.png')


def load_csv(path):
    if not os.path.exists(path):
        return None, None, None
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=str)
    if data.ndim == 0:
        return None, None, None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    iters = data[:, 0].astype(int)
    loss = data[:, 1].astype(float)
    ber = data[:, 2].astype(float) if data.shape[1] >= 3 else None
    return iters, loss, ber


def plot_stage(ax, path, label, color):
    iters, loss, _ = load_csv(path)
    if iters is not None and len(iters) > 0:
        ax.plot(iters, loss, color=color, linewidth=0.8, label=label)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -- Loss --
ax = axes[0]
plot_stage(ax, os.path.join(SCRIPT_DIR, 'metrics_stage1.csv'), 'Stage 1', '#1f77b4')
plot_stage(ax, os.path.join(SCRIPT_DIR, 'metrics_stage2.csv'), 'Stage 2', '#ff7f0e')
plot_stage(ax, os.path.join(SCRIPT_DIR, 'metrics_stage3.csv'), 'Stage 3', '#2ca02c')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.set_yscale('symlog', linthresh=1.0)
ax.grid(True, alpha=0.3)

# -- BER --
ax = axes[1]
for path, label, color, marker in [
    ('metrics_stage1.csv', 'Stage 1', '#1f77b4', 'o'),
    ('metrics_stage3.csv', 'Stage 3', '#2ca02c', 's'),
]:
    iters, _, ber = load_csv(os.path.join(SCRIPT_DIR, path))
    if iters is not None and ber is not None and len(iters) > 0:
        ax.plot(iters, ber, color=color, marker=marker, markersize=3,
                linewidth=0.8, label=label)
ax.set_xlabel('Iteration')
ax.set_ylabel('BER@20dB')
ax.set_title('Eval BER')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

fig.suptitle('qQ Model Training Curves', fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print(f'Plot saved to {OUTPUT}')
