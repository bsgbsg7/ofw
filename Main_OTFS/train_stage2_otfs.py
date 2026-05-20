"""
Stage 2: Fine-tune qQ_TV model from Stage 1 weights. Higher SNR.
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
import random
import logging
import numpy as np

tf.get_logger().setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

LEARNING_RATE = 0.0001
NUM_ITERS = 2000

weights_file_name = 'weights-qQ_Method_TV'
pretrained_weights_file_name = 'weights-qQ_Method_TV_initial'

model_train = qQ_MODEL_TV(training=True)
model_train(2, 40.0)
with open(pretrained_weights_file_name, 'rb') as f:
    weights = pickle.load(f)
    model_train.set_weights(weights)
print(f"Loaded pretrained weights from {pretrained_weights_file_name}")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

best_loss = np.inf
print(f"Stage 2 (OTFS): Fine-tuning for {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [20, 25]")

for i in range(NUM_ITERS):
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE * 256, random.uniform(20, EBN0_DB_MAX))
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))

    if i % 50 == 0:
        loss_val = float(loss)
        print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}")
        if loss_val < best_loss:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_loss = loss_val
            print(f"    -> Saved best weights (loss={best_loss:.4E})")

print(f"Stage 2 (OTFS) complete. Best loss: {best_loss:.4E}")
