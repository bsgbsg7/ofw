import sys
import pickle
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.E2EWL.E2EWL_Model import E2EWL_MODEL
from config import *
import keras
import tensorflow as tf
import random
import logging
# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)

# Training Param
LEARNING_RATE = 0.001
BATCH_SNR = EBNO_DB_for_training
MULTY_PATH_MODEL = True
MODEL_NEURAL_RECIVER = True

# Weights file name to save
weights_file_name = 'weights-E2EWL_MP' if MULTY_PATH_MODEL else 'weights-E2EWL_AWGN'

# Instantiating the model for training
model_train = E2EWL_MODEL(training=True, 
                          is_multypath=MULTY_PATH_MODEL, 
                          is_neural_reciver=MODEL_NEURAL_RECIVER)


# Optimizer
optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)

# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        # loss = model_train(BATCH_SIZE, BATCH_SNR) 
        loss = model_train(BATCH_SIZE*10, random.randint(20, 50)) 
    
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        model_train.visulaize_progress = True
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")

        # Save the weightsin a file
        weights = model_train.get_weights()
        with open(weights_file_name, 'wb') as f:
            pickle.dump(weights, f)
    else:
        model_train.visulaize_progress = False