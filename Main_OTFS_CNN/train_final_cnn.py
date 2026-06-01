"""
Train qQ_MODEL_TV_CNN with pre-trained Q-creator + MLP joint detector.

Strategy:
  1. Load pre-trained Q-creator weights from Main_OTFS/weights-qQ_Method_TV
  2. Use a lightweight MLP (2 Dense layers) as the joint detector
  3. Train end-to-end: Q-creator fine-tunes with low effective LR,
     MLP learns joint detection from scratch
  4. The pre-trained Q-creator already produces meaningful Q matrices,
     so the MLP can converge quickly to do joint detection.

Usage:
    python train_final_cnn.py
"""
import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.qQ_Method.qQ_Model_TV_CNN import qQ_MODEL_TV_CNN
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


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, alpha=0.0, warmup_steps=0, name=None):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = float(decay_steps)
        self.alpha = alpha
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup = self.initial_lr * (step_f / self.warmup_steps)
        decay_step = step_f - self.warmup_steps
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(np.pi) * decay_step / self.decay_steps))
        cosine_lr = self.alpha * self.initial_lr + (1.0 - self.alpha) * self.initial_lr * cosine
        return tf.where(step_f < self.warmup_steps, warmup, cosine_lr)

    def get_config(self):
        return {"initial_lr": self.initial_lr, "decay_steps": self.decay_steps,
                "alpha": self.alpha, "warmup_steps": self.warmup_steps}


# ---- Hyperparameters ----
LR_INITIAL = 1e-4          # low LR for fine-tuning
LR_ALPHA = 0.1             # final = 1e-5
LR_WARMUP = 100
NUM_ITERS = 5000

EFFECTIVE_BATCH = BATCH_SIZE * 32   # 320
TRAIN_SNR_MIN = 5.0         # start at higher SNR for faster learning
TRAIN_SNR_MAX = 20.0
EVAL_SNR = 15.0

weights_file_name = 'weights-qQ_Method_TV_CNN'

# ---- Model creation ----
model_train = qQ_MODEL_TV_CNN(training=True)
model_eval = qQ_MODEL_TV_CNN(training=False)
model_eval(2, 40.0)  # build eval model graph

# Build train model and load pre-trained weights
model_train(2, 40.0)

# Load transferred weights (run transfer_weights.py first to create this file)
# transfer_weights.py maps: PT[0:48]->CNN[0:48] (Q-creator), PT[48:76]->CNN[77:105] (uncertainty)
try:
    with open(weights_file_name, 'rb') as f:
        model_train.set_weights(pickle.load(f))
    print(f"Loaded transferred weights from {weights_file_name}")
    print("(Q-creator + uncertainty models pre-trained, CNNJointDetector random init)")
except FileNotFoundError:
    print(f"WARNING: {weights_file_name} not found. Run transfer_weights.py first.")
    print("Training from scratch.")


lr_schedule = WarmupCosineDecay(LR_INITIAL, NUM_ITERS, alpha=LR_ALPHA, warmup_steps=LR_WARMUP)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        loss = model_train(batch_size, ebno)
    grads = tape.gradient(loss, model_train.trainable_weights)
    grads, gnorm = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return loss, gnorm


best_ber = 1.0
patience_counter = 0

print(f"{'='*60}")
print(f"Fine-tuning qQ_MODEL_TV_CNN (MLP Joint Detector)")
print(f"{'='*60}")
print(f"Total iters: {NUM_ITERS}, Effective batch: {EFFECTIVE_BATCH}")
print(f"LR: {LR_INITIAL:.0E} -> {LR_INITIAL*LR_ALPHA:.0E} (cosine decay, warmup={LR_WARMUP})")
print(f"SNR range: [{TRAIN_SNR_MIN}, {TRAIN_SNR_MAX}] dB, Eval SNR: {EVAL_SNR} dB")
print(f"Pretrained via transfer_weights.py: {weights_file_name}")
print(f"{'='*60}")

for i in range(NUM_ITERS):
    loss, gnorm = train_step(
        tf.constant(EFFECTIVE_BATCH),
        tf.constant(TRAIN_SNR_MIN),
        tf.constant(TRAIN_SNR_MAX)
    )

    if i % 100 == 0:
        model_eval.set_weights(model_train.get_weights())
        total_ber = 0.0
        for _ in range(3):
            b, b_hat = model_eval(128, EVAL_SNR)
            total_ber += float(compute_ber(b, b_hat))
        avg_ber = total_ber / 3

        loss_val = float(loss)
        gnorm_val = float(gnorm)
        current_lr = float(lr_schedule(i).numpy())
        print(f"  Iter {i}/{NUM_ITERS}  Loss: {loss_val:.4E}  "
              f"BER@{EVAL_SNR:.0f}dB: {avg_ber:.5f}  "
              f"LR: {current_lr:.2E}  GradNorm: {gnorm_val:.2f}", flush=True)

        if avg_ber < best_ber:
            weights = model_train.get_weights()
            with open(weights_file_name, 'wb') as f:
                pickle.dump(weights, f)
            best_ber = avg_ber
            patience_counter = 0
            print(f"    -> Saved best weights (BER={best_ber:.5f})", flush=True)
        else:
            patience_counter += 1

        if patience_counter >= 15:
            print(f"Early stopping at iter {i}, best BER: {best_ber:.5f}")
            break

print(f"{'='*60}")
print(f"Training complete. Best BER@{EVAL_SNR:.0f}dB: {best_ber:.5f}")
print(f"Weights saved to: {weights_file_name}")
