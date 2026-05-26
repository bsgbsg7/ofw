"""
Stage 2: Fine-tune qQ model from Stage 1 weights.
Based on train.py with load_pretrained=True.
Uses smaller effective block (single symbol), lower LR, higher SNR.
"""
import sys
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import *
import keras
import tensorflow as tf
import random
import logging
import numpy as np
from sionna.phy.utils import compute_ber
from plot_training import save_plots

tf.get_logger().setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# Training Param - Stage 2: fine-tune with higher SNR
LEARNING_RATE = 0.0001
NUM_ITERS = 4000

# Weights file name to save
weights_file_name = 'weights-qQ_Method_Stage2'
pretrained_weights_file_name = 'weights-qQ_Method_initial'

# 创建训练和评估两个独立的模型实例
model_train = qQ_MODEL(training=True)
model_eval = qQ_MODEL(training=False)
model_eval(2, 40.0)  # 用虚拟输入构建 model_eval，确保权重可加载

# Load pretrained weights from stage 1
model_train(2, 40.0)  # build
with open(pretrained_weights_file_name, 'rb') as f:
    weights = pickle.load(f)
    model_train.set_weights(weights)
print(f"Loaded pretrained weights from {pretrained_weights_file_name}")

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# 早停相关状态
best_ber = 1.0
best_bce = float('inf')
best_loss = float('inf')
best_par = float('inf')
patience_counter = 0

# 记录训练过程中的指标
history = {
    'iter': [],
    'loss': [],
    'PAR': [],
    'BCE': [],
    'BER': [],
}
history_file = 'history_stage2.pkl'

print(f"Stage 2: Fine-tuning for {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [20, 25]")

for i in range(NUM_ITERS):
    with tf.GradientTape() as tape:
        total_loss, par, bce_loss = model_train(BATCH_SIZE * 256, random.uniform(20, EBN0_DB_MAX))
    grads = tape.gradient(total_loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))

    loss_val = float(total_loss)
    par_val = float(par)
    bce_val = float(bce_loss)

    # 每一步都记录训练指标
    history['iter'].append(i)
    history['loss'].append(loss_val)
    history['PAR'].append(par_val)
    history['BCE'].append(bce_val)

    # 每 50 步评估 BER、打印、保存权重、早停
    if i % 50 == 0:
        # 将训练权重同步到评估模型
        model_eval.set_weights(model_train.get_weights())

        # 在 20dB 处多次评估 BER 取平均
        total_ber = 0.0
        for _ in range(5):
            b, b_hat = model_eval(200, 20.0)
            total_ber += float(compute_ber(b, b_hat))
        avg_ber = total_ber / 5

        history['BER'].append(avg_ber)

        print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}  PAR: {par_val:.4f}  "
              f"BCE: {bce_val:.4f}  Eval BER@20dB: {avg_ber:.4f}")

        # 若 BER 和 BCE 同时改善才保存权重并重置 patience
        # if avg_ber < best_ber and bce_val < best_bce:
        if loss_val < best_loss:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_ber = avg_ber
            best_bce = bce_val
            best_loss = loss_val
            best_par = par_val
            patience_counter = 0
            print(f"    -> Saved best weights (BER={best_ber:.4f}, BCE={best_bce:.4f}, "
                  f"Loss={best_loss:.4E}, PAR={best_par:.4f})")
        else:
            patience_counter += 1

        # 连续 15 次评估（即 750 步）未同时改善则早停
        # if patience_counter >= 15:
        #     print(f"Early stopping at iter {i}, best BER: {best_ber:.4f}, best BCE: {best_bce:.4f}")
        #     break

print(f"Stage 2 complete. Best Loss: {best_loss:.4E}, Best PAR: {best_par:.4f}, Best BER@20dB: {best_ber:.4f}, Best BCE: {best_bce:.4f}")
print(f"Weights saved to: {weights_file_name}")

# 保存 history 并画图
with open(history_file, 'wb') as f:
    pickle.dump(history, f)
print(f"History data saved to: {history_file}")

save_plots(history, save_path='training_curves_stage2.png')
