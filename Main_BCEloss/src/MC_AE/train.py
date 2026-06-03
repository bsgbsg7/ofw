from MC_AE_Model import MC_AE_MODEL
import sys
import pickle
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import *
import keras
import tensorflow as tf

# Instantiating the end-to-end model for training
model_train = MC_AE_MODEL(training=True)

# Adam optimizer
optimizer = keras.optimizers.Adam()

# Training loop
for i in range(NUM_TRAINING_ITERATIONS):
    # Forward pass
    with tf.GradientTape() as tape:
        loss = model_train(BATCH_SIZE, EBNO_DB_for_training) 
    # Computing and applying gradients
    grads = tape.gradient(loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    # Print progress
    if i % 100 == 0:
        print(f"{i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss:.2E}", end="\r")

        # Save the weightsin a file
        weights = model_train.get_weights()
        with open('weights-MC_AE', 'wb') as f:
            pickle.dump(weights, f)