import sys
import pickle
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import *
import keras
import tensorflow as tf
import random
import logging
import numpy as np
# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)

# Training Param
LEARNING_RATE = 0.001
BATCH_SNR = EBNO_DB_for_training
load_pretrained = False

# Weights file name to save
weights_file_name = 'weights-qQ_Method'

# Instantiating the model for training
model_train = qQ_MODEL(training=True)

# load pretrained
pretrained_weights_file_name = 'weights-qQ_Method_initial'
if load_pretrained:
    model_train(tf.constant(2),tf.constant(40))  # call the model once to build so weights could be loaded
    with open(pretrained_weights_file_name, 'rb') as f:
        weights = pickle.load(f)
        model_train.set_weights(weights)

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    # Generate a random EbNo for this step inside the graph
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    
    with tf.GradientTape() as tape:
        # Forward pass
        loss, _, _ = model_train(batch_size, ebno)

    # Backpropagation
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return loss

best_loss = np.inf
# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward - Backward pass
    loss = train_step(
        tf.constant(BATCH_SIZE * 256), 
        tf.constant(float(EBN0_DB_MIN + 10)), 
        tf.constant(float(EBN0_DB_MAX))
    )

    # Logging and Saving (Keep this in Python/Eager mode)
    if i % 100 == 0:
        # model_train.visulaize_progress = True 
        print(f"Iteration {i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}")
        if loss < best_loss:
            # Save the weightsin a file
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_loss = loss
    else:
        model_train.visulaize_progress = False