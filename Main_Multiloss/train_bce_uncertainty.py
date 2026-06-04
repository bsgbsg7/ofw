"""
4-Loss Uncertainty-Weighted Multi-Task Training (loss.md design).

Losses:
  L_BCE  — Binary Cross-Entropy (通信可靠性)
  L_PAPR — Peak-to-Average Power Ratio (峰均比)
  L_OOB  — Out-of-Band Emission (带外泄漏)
  L_AF   — Ambiguity Function Shape (模糊函数整形)

  L_total = Σ_i [ exp(-logσ²_i) · L_i · scale_i + logσ²_i ]

UncertaintyModel_4D predicts logσ²_i from RMS delay spread,
enabling automatic waveform adaptation across channel conditions.

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

weights_file_name = 'weights-qQ_Method_4Loss'

model_train = qQ_MODEL(training=True)
model_eval = qQ_MODEL(training=False)

# Build both models
model_train(2, 40.0)
model_eval(2, 40.0)
print(f"Model built from scratch (random init)")
print(f"Loss: Kendall Uncertainty-Weighted 4-Loss (BCE + PAPR + OOB + AF)")
print(f"Channel: TDL_RandomDS (10ns - 600ns, no Doppler)")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        (total_loss, bce, papr, oob, af,
         ls_bce, ls_papr, ls_oob, ls_af) = model_train(batch_size, ebno)
    grads = tape.gradient(total_loss, model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return total_loss, bce, papr, oob, af, ls_bce, ls_papr, ls_oob, ls_af

best_ber = 1.0
print(f"Training for up to {NUM_ITERS} iterations")
print(f"Batch size: {BATCH_SIZE * 256}, LR: {LEARNING_RATE}, SNR range: [{EBN0_DB_MIN + 10}, {EBN0_DB_MAX}]")
print(f"GPU: 1")
print(f"{'Iter':>6} {'Loss':>10} {'BCE':>10} {'PAPR':>10} {'OOB':>10} {'AF':>10} "
      f"{'w_bce':>8} {'w_papr':>8} {'w_oob':>8} {'w_af':>8} {'BER':>10}")

for i in range(NUM_ITERS):
    loss, bce, papr, oob, af, ls_b, ls_p, ls_o, ls_a = train_step(
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

        # Extract scalar values
        def f(x): return float(x)

        # Compute uncertainty weights (precision)
        w_bce  = np.exp(-f(ls_b))
        w_papr = np.exp(-f(ls_p))
        w_oob  = np.exp(-f(ls_o))
        w_af   = np.exp(-f(ls_a))

        print(f"  {i:4d}  {f(loss):10.4E} {f(bce):10.4E} {f(papr):10.4E} "
              f"{f(oob):10.4E} {f(af):10.4E} "
              f"{w_bce:8.3f} {w_papr:8.3f} {w_oob:8.3f} {w_af:8.3f} "
              f"{avg_ber:10.5f}", flush=True)

        if avg_ber < best_ber:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_ber = avg_ber
            print(f"    -> Saved best weights (BER={best_ber:.5f})", flush=True)

print(f"\n4-Loss uncertainty-weighted training complete.")
print(f"Best BER@20dB: {best_ber:.5f}")
print(f"Weights saved to: {weights_file_name}")
print(f"\nNow run analysis to visualize learned waveforms.")
