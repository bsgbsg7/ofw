"""
OTFS 训练 — Kendall 不确定性加权 BCE+PAPR + Cosine Decay LR
==========================================================
低 SNR [0,15]dB + 不确定性模型 (log_sigma ∈ [-3,3]):
  → 平坦信道: 不确定性 → 高 PAPR 权重 → TDM
  → 多径低速: 不确定性 → 低 PAPR 权重 → OFDM
  → 多径高速: BCE 探索 → OTFS
"""
import sys, pickle, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import tensorflow as tf
import keras
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

from config import *
from src.qQ_Method.qQ_Model_TV import qQ_MODEL_TV


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, alpha=0.0, warmup_steps=0):
        super().__init__()
        self.initial_lr = initial_lr; self.decay_steps = float(decay_steps)
        self.alpha = alpha; self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup = self.initial_lr * (step_f / self.warmup_steps)
        decay_step = step_f - self.warmup_steps
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(np.pi) * decay_step / self.decay_steps))
        lr = self.alpha * self.initial_lr + (1.0 - self.alpha) * self.initial_lr * cosine
        return tf.where(step_f < self.warmup_steps, warmup, lr)

    def get_config(self):
        return {"initial_lr": self.initial_lr, "decay_steps": self.decay_steps,
                "alpha": self.alpha, "warmup_steps": self.warmup_steps}


model_train = qQ_MODEL_TV(training=True)
model_eval = qQ_MODEL_TV(training=False)
model_eval(2, 40.0)

LR_INITIAL = 0.0005; LR_ALPHA = 0.1; LR_WARMUP = 1000
NUM_ITERS = 15000
EFFECTIVE_BATCH = BATCH_SIZE * 8
weights_file = 'weights-qQ_Method_TV_64_10M'

# Train from scratch — uncertainty model learns to balance BCE/PAPR jointly
print("Training from scratch: Kendall uncertainty + low SNR.")

lr_schedule = WarmupCosineDecay(LR_INITIAL, NUM_ITERS, alpha=LR_ALPHA, warmup_steps=LR_WARMUP)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# Use first GPU explicitly
with tf.device('/GPU:0'):
    model_train(2, 40.0)

@tf.function
def train_step(batch_size, ebno_min, ebno_max):
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        loss = model_train(batch_size, ebno)
    grads = tape.gradient(loss, model_train.trainable_weights)
    grads, gnorm = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
    return loss, gnorm

best_ber = 1.0; patience = 0
TRAIN_SNR_MIN = 0.0   # lower SNR → BER differences amplified
TRAIN_SNR_MAX = 15.0
EVAL_SNR = 15.0       # eval at training SNR ceiling
print(f"\n{'='*60}")
print(f"OTFS Training — Kendall uncertainty BCE+PAPR (log_sigma ∈ [-3,3])")
print(f"  LR: {LR_INITIAL:.0E}→{LR_INITIAL*LR_ALPHA:.0E}, Iters: {NUM_ITERS}")
print(f"  Delay: [{DELAY_SPREAD_MIN*1e9:.0f}-{DELAY_SPREAD_MAX*1e9:.0f}]ns")
print(f"  Speed: [{SPEED_MIN}-{SPEED_MAX}]m/s, SNR: [{TRAIN_SNR_MIN}-{TRAIN_SNR_MAX}]dB")
print(f"  PAPR: ON (Kendall uncertainty, bounds [-3,3]), Batch: {EFFECTIVE_BATCH}")
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
        lr = float(lr_schedule(i).numpy())
        print(f"  Iter {i}/{NUM_ITERS}  Loss: {float(loss):.4E}  "
              f"BER@{EVAL_SNR:.0f}dB: {avg_ber:.5f}  LR: {lr:.2E}  GradNorm: {float(gnorm):.2f}", flush=True)

        if avg_ber < best_ber:
            with open(weights_file, 'wb') as f:
                pickle.dump(model_train.get_weights(), f)
            best_ber = avg_ber; patience = 0
            print(f"    -> Saved (BER={best_ber:.5f})", flush=True)
        else:
            patience += 1
        if patience >= 50:
            print(f"Early stop at iter {i}, best BER: {best_ber:.5f}")
            break

print(f"\nTraining done. Best BER@{EVAL_SNR:.0f}dB: {best_ber:.5f}")
print(f"Weights saved to: {weights_file}")
