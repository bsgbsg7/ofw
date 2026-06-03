import sys
import pickle
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.RQ_Method.RQ_Model import RQ_MODEL
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


# Weights file name to save
weights_file_name = 'weights-RQ_Method'

# Instantiating the model for training
model_train = RQ_MODEL(training=True)

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)

best_loss = np.inf
# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        # loss = model_train(BATCH_SIZE, BATCH_SNR) 
        loss = model_train(BATCH_SIZE*32, random.uniform(EBN0_DB_MIN, EBN0_DB_MAX)) 
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        model_train.visulaize_progress = True
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")
        if loss < best_loss:
            # Save the weightsin a file
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_loss = loss
    else:
        model_train.visulaize_progress = False