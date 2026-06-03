import sys
import pickle
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import *
import keras
import tensorflow as tf
import numpy as np
import logging

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)

# Setup Strategy
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) >= 2:
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    print(f"Running on {len(gpus)} GPUs")
else:
    strategy = tf.distribute.get_strategy()

# Training Params
LEARNING_RATE = 0.00001 #0.001 0.0001 0.00001
BATCH_SIZE_PER_DEVICE = BATCH_SIZE * 700
weights_file_name = 'weights-qQ_Method'
load_pretrained = True
pretrained_weights_file_name = 'weights-qQ_Method'

# Dummy Instantiation using Dist strategy
with strategy.scope():
    model_train = qQ_MODEL(training=True)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Load pretrained weights inside scope if needed
    if load_pretrained:
        # Dummy call to build variables
        model_train(tf.constant(2), tf.constant(40.0))
        with open(pretrained_weights_file_name, 'rb') as f:
            weights = pickle.load(f)
            model_train.set_weights(weights)

# Define the distributed training step
@tf.function
def distributed_train_step(batch_size, ebno_min, ebno_max):
    def step_fn(b_size, e_min, e_max):
        ebno = tf.random.uniform([], e_min, e_max)
        with tf.GradientTape() as tape:
            loss = model_train(b_size, ebno)
            # Scale loss by number of replicas to keep gradients stable
            scaled_loss = loss / strategy.num_replicas_in_sync
            
        grads = tape.gradient(scaled_loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
        return loss

    # Distribute the computation
    per_replica_losses = strategy.run(step_fn, args=(batch_size, ebno_min, ebno_max))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

best_loss = np.inf
# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    loss = distributed_train_step(
        tf.constant(BATCH_SIZE_PER_DEVICE), 
        tf.constant(float(EBN0_DB_MIN + 20)), 
        tf.constant(float(EBN0_DB_MAX))
    )

    if i % 100 == 0:
        print(f"Iteration {i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}")
        if loss < best_loss:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_loss = loss