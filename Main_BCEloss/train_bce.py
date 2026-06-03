"""
BCE-only Training: Train from scratch using only BCE loss.
Uses GPU 1, lower LR, and evaluation-driven weight saving.
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
NUM_ITERS = 20000

weights_file_name = 'weights-qQ_Method_BCE'

model_train = qQ_MODEL(training=True)
model_eval = qQ_MODEL(training=False)

# Build both models
model_train(2, 40.0)
model_eval(2, 40.0)
print("Model built from scratch (random init) — no pretrained weights")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        total_loss, bce_loss = model_train(batch_size, ebno)
    grads = tape.gradient(total_loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return total_loss, bce_loss

best_ber = 0.01030
patience_counter = 0
print(f"BCE-only Training for up to {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [10, 25]")
print(f"GPU: 1")

for i in range(NUM_ITERS):
    loss, bce = train_step(
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
        print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}  BCE: {bce_val:.4E}  Eval BER@20dB: {avg_ber:.5f}", flush=True)

        if avg_ber < best_ber:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_ber = avg_ber
            patience_counter = 0
            print(f"    -> Saved best weights (BER={best_ber:.5f})", flush=True)
        else:
            patience_counter += 1

print(f"BCE-only training complete. Best BER@20dB: {best_ber:.5f}")
print(f"Weights saved to: {weights_file_name}")
