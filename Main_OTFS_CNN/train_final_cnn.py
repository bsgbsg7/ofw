"""
Two-phase training with MSE supervision for MLP equalizer.

Phase 1: Freeze Q-creator, train MLP with MSE loss
  - MLP learns to predict true transmitted QAM symbols from Q-demodulated symbols
  - MSE is smooth and convex → fast convergence
  - Higher SNR [10,25]dB for cleaner supervision signal

Phase 2: Joint fine-tuning with BCE loss
  - Unfreeze Q-creator, all layers train with low LR
  - BCE through Demapper for BER optimization
"""
import sys, pickle, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.qQ_Method.qQ_Model_TV_CNN import qQ_MODEL_TV_CNN, MLPEqualizer
from config import *
import keras, tensorflow as tf, logging, numpy as np
from sionna.phy.utils import ebnodb2no, compute_ber, hard_decisions
from sionna.phy.channel import cir_to_time_channel

tf.get_logger().setLevel(logging.ERROR)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, alpha=0.0, warmup_steps=0):
        super().__init__()
        self.initial_lr, self.decay_steps = initial_lr, float(decay_steps)
        self.alpha, self.warmup_steps = alpha, float(warmup_steps)
    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup = self.initial_lr * (step_f / self.warmup_steps)
        decay_step = step_f - self.warmup_steps
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(np.pi) * decay_step / self.decay_steps))
        return tf.where(step_f < self.warmup_steps, warmup,
                        self.alpha * self.initial_lr + (1.0 - self.alpha) * self.initial_lr * cosine)


EFFECTIVE_BATCH = BATCH_SIZE * 32
EVAL_SNR = 20.0
weights_file = 'weights-qQ_Method_TV_CNN'

# ---- Create models ----
model_train = qQ_MODEL_TV_CNN(training=True)
model_eval = qQ_MODEL_TV_CNN(training=False)
model_train(2, 40.0); model_eval(2, 40.0)
try:
    with open(weights_file, 'rb') as f:
        model_train.set_weights(pickle.load(f))
    print(f"Loaded: {weights_file}")
except FileNotFoundError:
    print("WARNING: run transfer_weights.py first")

# Identify vars
mlp_vars = model_train._mlp_equalizer.trainable_variables
mlp_var_refs = {v.ref() for v in mlp_vars}
all_vars = model_train.trainable_variables

def evaluate():
    model_eval.set_weights(model_train.get_weights())
    total_ber = 0.0
    for _ in range(3):
        b, b_hat = model_eval(128, EVAL_SNR)
        total_ber += float(compute_ber(b, b_hat))
    return total_ber / 3

best_ber = 1.0
patience = 0

# ================================================================
# Phase 1: Freeze Q-creator, train MLP with MSE
# ================================================================
PHASE1_ITERS = 1500
print(f"\n{'='*60}")
print(f"PHASE 1: Train MLP equalizer with MSE loss ({PHASE1_ITERS} iters)")
print(f"{'='*60}")

lr1 = WarmupCosineDecay(1e-3, PHASE1_ITERS, alpha=0.1, warmup_steps=100)
opt1 = keras.optimizers.Adam(learning_rate=lr1)

@tf.function
def train_step_p1(batch_size, ebno_min, ebno_max):
    """MSE loss: MLP learns to predict true QAM symbols."""
    ebno = tf.random.uniform([], ebno_min, ebno_max)

    # Forward pass through TX + channel (no Demapper, just get equalized symbols)
    with tf.GradientTape() as tape:
        # Run the model's internal forward pass to get intermediate values
        no = ebnodb2no(ebno, model_train._num_bits_per_symbol, model_train._coderate,
                       model_train._rg)
        b = model_train._binary_source([batch_size, 1, model_train._num_streams_per_tx,
                                         model_train._n])
        x_symbols = model_train._mapper(b)  # (B,1,1,64) — true QAM symbols
        x_rg = model_train._rg_mapper(x_symbols)

        # Channel
        a, tau = model_train._channel_model(batch_size,
                                            model_train._rg.num_time_samples + model_train._l_tot - 1,
                                            model_train._rg.bandwidth)
        h_time = cir_to_time_channel(model_train._rg.bandwidth, a, tau,
                                     l_min=model_train._l_min, l_max=model_train._l_max,
                                     normalize=True)

        # Q creation
        h_2d = h_time[:, 0, 0, 0, 0, :, :]
        total_time = tf.shape(h_2d)[1]
        stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
        indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
        pilots = tf.gather(h_2d, indices, axis=1)
        pilots = tf.transpose(pilots, [0, 2, 1])
        Q, q = model_train._qQ_creator_layer(pilots, training=False)  # frozen

        x_time = model_train._Q_modulator(Q, x_rg)
        y_time = model_train._channel_time(x_time, h_time, no)

        # Q^H Demodulation
        r_freq = model_train._Q_demodulator(Q, y_time)
        r_freq = r_freq[:, :, :, 1:, :]            # (B,1,1,2,32)
        # Apply q (per-subcarrier scaling) — essential!
        q_bc = q[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        q_bc = tf.tile(q_bc, [1, 1, 1, tf.shape(r_freq)[-2], 1])
        r_freq = r_freq * q_bc
        r_freq_data = r_freq[:, 0, 0, :, :]        # (B,2,32) complex

        # Flatten
        r_flat = tf.concat([
            tf.reshape(tf.math.real(r_freq_data), [batch_size, -1]),
            tf.reshape(tf.math.imag(r_freq_data), [batch_size, -1])
        ], axis=-1)  # (B, 128)

        # MLP equalizer
        r_eq_flat = model_train._mlp_equalizer(r_flat, training=True)  # (B, 128)
        r_eq_real = r_eq_flat[:, :64]
        r_eq_imag = r_eq_flat[:, 64:]
        r_eq = tf.complex(r_eq_real, r_eq_imag)     # (B, 64)

        # MSE loss against true symbols
        x_true = tf.squeeze(x_symbols, axis=[1, 2])  # (B, 64) complex
        mse = tf.reduce_mean(tf.abs(r_eq - x_true) ** 2)

    # Only apply grads to MLP vars
    grads = tape.gradient(mse, all_vars)
    mlp_grads = [g for g, v in zip(grads, all_vars) if v.ref() in mlp_var_refs]
    mlp_grads, gnorm = tf.clip_by_global_norm(mlp_grads, 5.0)
    opt1.apply_gradients(zip(mlp_grads, mlp_vars))
    return mse, gnorm

for i in range(PHASE1_ITERS):
    mse, gnorm = train_step_p1(tf.constant(EFFECTIVE_BATCH),
                               tf.constant(10.0), tf.constant(25.0))
    if i % 100 == 0:
        avg_ber = evaluate()
        print(f"  P1 Iter {i}/{PHASE1_ITERS}  MSE: {float(mse):.4E}  "
              f"BER@{EVAL_SNR:.0f}dB: {avg_ber:.5f}  "
              f"LR: {float(lr1(i)):.2E}  GN: {float(gnorm):.2f}", flush=True)
        if avg_ber < best_ber:
            best_ber = avg_ber; patience = 0
            with open(weights_file, 'wb') as f: pickle.dump(model_train.get_weights(), f)
            print(f"    -> Saved (BER={best_ber:.5f})", flush=True)
        else:
            patience += 1
        if patience >= 12: break

print(f"Phase 1 done. Best BER: {best_ber:.5f}")

# ================================================================
# Phase 2: Joint fine-tuning with BCE (original model.call())
# ================================================================
PHASE2_ITERS = 2000
print(f"\n{'='*60}")
print(f"PHASE 2: Joint BCE fine-tuning ({PHASE2_ITERS} iters)")
print(f"{'='*60}")

lr2 = WarmupCosineDecay(1e-4, PHASE2_ITERS, alpha=0.5, warmup_steps=0)
opt2 = keras.optimizers.Adam(learning_rate=lr2)

@tf.function
def train_step_p2(batch_size, ebno_min, ebno_max):
    """BCE loss through full model.call()."""
    ebno = tf.random.uniform([], ebno_min, ebno_max)
    with tf.GradientTape() as tape:
        loss = model_train(batch_size, ebno)  # uses BCE through Demapper
    grads = tape.gradient(loss, all_vars)     # all vars updated
    grads, gnorm = tf.clip_by_global_norm(grads, 5.0)
    opt2.apply_gradients(zip(grads, all_vars))
    return loss, gnorm

for i in range(PHASE2_ITERS):
    loss, gnorm = train_step_p2(tf.constant(EFFECTIVE_BATCH),
                                tf.constant(0.0), tf.constant(25.0))
    if i % 100 == 0:
        avg_ber = evaluate()
        print(f"  P2 Iter {i}/{PHASE2_ITERS}  Loss: {float(loss):.4E}  "
              f"BER@{EVAL_SNR:.0f}dB: {avg_ber:.5f}  "
              f"LR: {float(lr2(i)):.2E}  GN: {float(gnorm):.2f}", flush=True)
        if avg_ber < best_ber:
            best_ber = avg_ber; patience = 0
            with open(weights_file, 'wb') as f: pickle.dump(model_train.get_weights(), f)
            print(f"    -> Saved (BER={best_ber:.5f})", flush=True)
        else:
            patience += 1
        if patience >= 15: break

print(f"\n{'='*60}")
print(f"Training complete. Best BER@{EVAL_SNR:.0f}dB: {best_ber:.5f}")
print(f"Weights: {weights_file}")

""
