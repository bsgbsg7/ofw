"""
训练曲线绘制工具 — 可独立运行，也可被训练脚本 import 调用。
用法：
    python plot_training.py history_data.pkl              # 从命令行指定 history 文件
    python plot_training.py history_data.pkl --save my.png  # 指定输出路径
"""

import pickle
import argparse
import matplotlib.pyplot as plt


def save_plots(history, save_path='training_curves_stage1.png'):
    """根据 history 字典绘制并保存训练曲线。"""
    if len(history['iter']) == 0:
        print("No training data collected, skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    iters = history['iter']

    # 图 1：Total Loss（线性坐标，因为可能为负）
    ax1 = axes[0, 0]
    ax1.plot(iters, history['loss'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss')
    ax1.grid(True, alpha=0.3)

    # 图 2：BCE Loss（对数坐标系）
    ax2 = axes[0, 1]
    ax2.plot(iters, history['BCE'], 'c-', linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('BCE Loss')
    ax2.set_title('BCE Loss')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 图 3：PAPR
    ax3 = axes[1, 0]
    ax3.plot(iters, history['PAR'], 'r-', linewidth=1.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('PAPR Loss')
    ax3.set_title('PAPR')
    ax3.grid(True, alpha=0.3)

    # 图 4：BER（对数坐标系）
    ax4 = axes[1, 1]
    ber_iters = [i * 100 for i in range(len(history['BER']))]
    ax4.plot(ber_iters, history['BER'], 'g-', linewidth=1.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('BER')
    ax4.set_title('Eval BER @ 20dB')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from saved history.')
    parser.add_argument('history_file', nargs='?', default='history_stage1.pkl',
                        help='Path to the .pkl history file (default: history_stage1.pkl)')
    parser.add_argument('--save', '-s', default=None,
                        help='Output image path (default: <history_file>_curves.png)')
    args = parser.parse_args()

    with open(args.history_file, 'rb') as f:
        history = pickle.load(f)

    save_path = args.save or args.history_file.replace('.pkl', '_curves.png')
    save_plots(history, save_path)


if __name__ == '__main__':
    main()
