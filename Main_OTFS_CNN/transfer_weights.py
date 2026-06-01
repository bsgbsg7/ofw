#!/usr/bin/env python3
"""
Transfer pre-trained Q-creator + uncertainty model weights to CNN model.

Weight mapping (verified by shape comparison):
  PT[ 0:48] → CNN[ 0:48]  qQ_creator_conv_gru          (48 weights, identical)
  PT[48:76] → CNN[59:87]  UncertaintyModel_2D + _1D    (28 weights, identical)
  CNN[48:59]               MLPEqualizer                 (11 weights, NEW — zero init)

Run once before training:
    python transfer_weights.py
"""
import pickle, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.qQ_Method.qQ_Model_TV_CNN import qQ_MODEL_TV_CNN
import sionna.phy as sn

sn.config.seed = 42

# Load pre-trained weights
with open('../Main_OTFS/weights-qQ_Method_TV', 'rb') as f:
    pt_weights = pickle.load(f)
print(f"Loaded pre-trained weights: {len(pt_weights)} arrays")

# Create CNN model and get fresh weights
model = qQ_MODEL_TV_CNN(training=False)
model(2, 40.0)  # build
cnn_weights = model.get_weights()
print(f"CNN model weights: {len(cnn_weights)} arrays")

# Transfer (indices verified by shape comparison)
cnn_weights = list(cnn_weights)  # make mutable (87 weights total)
cnn_weights[0:48] = pt_weights[0:48]        # Q-creator (48 weights)
# CNN[48:59] = MLPEqualizer (11 weights, keep zero init)
cnn_weights[59:87] = pt_weights[48:76]      # UncertaintyModel_2D + _1D (28 weights)

model.set_weights(cnn_weights)
print("Weight transfer complete.")

# Save
with open('weights-qQ_Method_TV_CNN', 'wb') as f:
    pickle.dump(model.get_weights(), f)
print("Saved: weights-qQ_Method_TV_CNN")

# Quick verification
import tensorflow as tf
b, b_hat = model(4, 20.0)
ber = float(tf.reduce_mean(tf.cast(tf.not_equal(b, b_hat), tf.float32)))
print(f"Quick test BER@20dB: {ber:.5f}")
print("Done.")
