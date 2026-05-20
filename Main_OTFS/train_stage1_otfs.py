"""
Stage 1: Train qQ_TV model from scratch with time-varying channel.
"""
import sys
import pickle
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV
from config import *
import keras
import tensorflow as tf
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

LEARNING_RATE = 0.001
NUM_ITERS = 5000

weights_file_name = 'weights-qQ_Method_TV_initial'

model_train = qQ_MODEL_TV(training=True)
model_eval = qQ_MODEL_TV(training=False)
model_eval(2, 40.0)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        loss = model_train(batch_size, ebno)
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return loss

best_ber = 1.0
patience_counter = 0
print(f"Stage 1 (OTFS): Training from scratch for up to {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [{EBN0_DB_MIN+10}, {EBN0_DB_MAX}]")
print(f"Speed range: [{SPEED_MIN}, {SPEED_MAX}] m/s")

for i in range(NUM_ITERS):
    loss = train_step(
        tf.constant(BATCH_SIZE * 256),
        tf.constant(float(EBN0_DB_MIN + 10)),
        tf.constant(float(EBN0_DB_MAX))
    )

    if i % 100 == 0:
        model_eval.set_weights(model_train.get_weights())
        total_ber = 0.0
        for _ in range(5):
            b, b_hat = model_eval(200, 20.0)
            total_ber += float(compute_ber(b, b_hat))
        avg_ber = total_ber / 5

        loss_val = float(loss)
        print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}  Eval BER@20dB: {avg_ber:.4f}")

        if avg_ber < best_ber:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_ber = avg_ber
            patience_counter = 0
            print(f"    -> Saved best weights (BER={best_ber:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= 15:
            print(f"Early stopping at iter {i}, best BER: {best_ber:.4f}")
            break

print(f"Stage 1 (OTFS) complete. Best BER@20dB: {best_ber:.4f}")
print(f"Weights saved to: {weights_file_name}")
