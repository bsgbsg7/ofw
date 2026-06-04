"""
Uncertainty-Guided Multi-Task Training.
Loss = BCE + lambda_papr(rms_ds) * PAPR_SCALE * PAPR

The Uncertainty Network predicts an adaptive PAPR weight and threshold
from RMS delay spread, allowing the model to automatically adjust
the PAPR constraint strength based on channel conditions.

Train on TDL_RandomDS (delay spread 10ns-600ns, no Doppler).
Uses GPU 1.
"""
import sys
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

tf.get_logger().setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

LEARNING_RATE = 0.0001
NUM_ITERS = 10000

weights_file_name = 'weights-qQ_Method_BCE_Uncertainty'

model_train = qQ_MODEL(training=True)
model_eval = qQ_MODEL(training=False)

# Build both models
model_train(2, 40.0)
model_eval(2, 40.0)
print(f"Model built from scratch (random init)")
print(f"Loss: Uncertainty-weighted multi-task (BCE + PAPR)")
print(f"Channel: TDL_RandomDS (10ns - 600ns, no Doppler)")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        total_loss, bce_loss, papr_loss, lambda_papr = model_train(batch_size, ebno)
    grads = tape.gradient(total_loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return total_loss, bce_loss, papr_loss, lambda_papr

best_ber = 1.0
print(f"Training for up to {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [{EBN0_DB_MIN + 10}, {EBN0_DB_MAX}]")
print(f"GPU: 1")

for i in range(NUM_ITERS):
    loss, bce, papr, lam = train_step(
        tf.constant(BATCH_SIZE * 256),
        tf.constant(float(EBN0_DB_MIN + 10)),
        tf.constant(float(EBN0_DB_MAX))
    )

    if i % 200 == 0:
        model_eval.set_weights(model_train.get_weights())
        total_ber = 0.0
        for _ in range(5):
            b, b_hat = model_eval(300, 20.0)
            total_ber += float(compute_ber(b, b_hat))
        avg_ber = total_ber / 5

        loss_val = float(loss)
        bce_val = float(bce)
        papr_val = float(papr)
        lam_val = float(lam)

        print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}  BCE: {bce_val:.4E}  "
              f"PAPR: {papr_val:.4E}  λ_papr: {lam_val:.5f}  "
              f"BER@20dB: {avg_ber:.5f}", flush=True)

        if avg_ber < best_ber:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_ber = avg_ber
            print(f"    -> Saved best weights (BER={best_ber:.5f})", flush=True)

print(f"\nUncertainty-weighted multi-task training complete.")
print(f"Best BER@20dB: {best_ber:.5f}")
print(f"Weights saved to: {weights_file_name}")
print(f"\nNow run analysis script to visualize learned waveforms.")
