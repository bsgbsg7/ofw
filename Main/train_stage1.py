"""
Stage 1: 从头训练 qQ 模型，基于 BER 进行 checkpoint 保存和早停。
训练结束后绘制 loss、PAPR、BER 下降曲线。
"""

import sys
import pickle
import os

# 将项目根目录和 src 目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.qQ_Method.qQ_Model import qQ_MODEL
from config import *
import keras
import tensorflow as tf
import logging
import numpy as np
from sionna.phy.utils import compute_ber
from plot_training import save_plots
from tqdm import tqdm

# 抑制 TensorFlow 日志输出，仅显示 ERROR 级别
tf.get_logger().setLevel(logging.ERROR)

# GPU 内存按需增长，避免一次性占满全部显存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# 训练超参数
LEARNING_RATE = 0.0005
NUM_ITERS = 5000

# 第一阶段权重保存文件名（后续阶段可加载此权重继续训练）
weights_file_name = 'weights-qQ_Method_initial'

# 创建训练和评估两个独立的模型实例
model_train = qQ_MODEL(training=True)   # 训练模式（含正则化等）
model_eval = qQ_MODEL(training=False)   # 评估模式（无随机性）
model_eval(2, 40.0)  # 用虚拟输入构建 model_eval，确保权重可加载

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    """单步训练：在 [ebno_min, ebno_max] 范围内随机采样 SNR 进行前向+反向传播。
    返回 (loss, PAR, BCE loss)。"""
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        total_loss, par, bce_loss = model_train(batch_size, ebno)
    grads = tape.gradient(total_loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return total_loss, par, bce_loss

# 早停相关状态
best_ber = 1.0           # 当前最佳 BER（初始化为最差值）
best_bce = float('inf')  # 当前最佳 BCE loss（初始化为无穷大）
patience_counter = 0     # 连续未同时改善的评估次数

# 记录训练过程中的指标
history = {
    'iter': [],       # 记录时的迭代步数
    'loss': [],       # 总 loss
    'PAR': [],        # PAPR loss
    'BCE': [],        # BCE loss
    'BER': [],        # 评估 BER@20dB
}


history_file = 'history_stage1.pkl'

print(f"Stage 1: Training from scratch for up to {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [{EBN0_DB_MIN+10}, {EBN0_DB_MAX}]")

try:
    for i in range(NUM_ITERS):
        # 单步训练
        loss, par, bce = train_step(
            tf.constant(BATCH_SIZE * 256),
            tf.constant(float(EBN0_DB_MIN + 10)),
            tf.constant(float(EBN0_DB_MAX))
        )

        loss_val = float(loss)
        par_val = float(par)
        bce_val = float(bce)

        # 每一步都记录训练指标
        history['iter'].append(i)
        history['loss'].append(loss_val)
        history['PAR'].append(par_val)
        history['BCE'].append(bce_val)

        # 每 100 步评估 BER、打印、保存权重、早停
        if i % 100 == 0:
            # 将训练权重同步到评估模型
            model_eval.set_weights(model_train.get_weights())

            # 在 20dB 处多次评估 BER 取平均，减少方差
            total_ber = 0.0
            for _ in range(5):
                b, b_hat = model_eval(200, 20.0)
                total_ber += float(compute_ber(b, b_hat))
            avg_ber = total_ber / 5

            # BER 仅在评估节点记录（每 100 步）
            history['BER'].append(avg_ber)

            print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}  PAR: {par_val:.4f}  "
                  f"BCE: {bce_val:.4f}  Eval BER@20dB: {avg_ber:.4f}")

            # 若 BER 和 BCE 同时改善才保存权重并重置 patience
            if avg_ber < best_ber and bce_val < best_bce:
                weights = model_train.get_weights()
                with open(weights_file_name, 'wb') as f:
                    pickle.dump(weights, f)
                best_ber = avg_ber
                best_bce = bce_val
                patience_counter = 0
                print(f"    -> Saved best weights (BER={best_ber:.4f}, BCE={best_bce:.4f})")
            else:
                patience_counter += 1

            # 连续 15 次评估（即 1500 步）未同时改善则早停
            if patience_counter >= 15:
                print(f"Early stopping at iter {i}, best BER: {best_ber:.4f}, best BCE: {best_bce:.4f}")
                break

    print(f"Stage 1 complete. Best BER@20dB: {best_ber:.4f}")
    print(f"Weights saved to: {weights_file_name}")

except KeyboardInterrupt:
    print(f"\nTraining interrupted at iter {history['iter'][-1] if history['iter'] else 'N/A'}. "
          f"Saving plots with collected data...")

finally:
    # 保存 history 数据，方便训练结束后重新画图
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    print(f"History data saved to: {history_file}")

    # 绘制并保存训练曲线
    save_plots(history)
