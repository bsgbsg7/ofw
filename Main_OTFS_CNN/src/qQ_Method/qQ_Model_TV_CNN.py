"""
qQ Model with Time-Varying channel + MLP joint equalizer.

The MLP equalizer replaces the one-tap equalizer (r_freq * q).
It processes ALL 64 QAM symbols jointly, enabling cross-subcarrier
and cross-symbol interference cancellation (essential for OTFS).

Sionna Demapper is RETAINED for constellation-aware LLR computation.

Architecture:
  y_time -> Q^H -> r_freq (2x32) -> flatten -> MLP -> reshape
         -> Demapper -> LLR -> BCE loss
"""
import tensorflow as tf
import keras
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import *
from channel_tv import channel_model
import numpy as np
from sionna.phy.mimo.stream_management import StreamManagement
from sionna.phy.ofdm.resource_grid import ResourceGrid
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.channel import time_lag_discrete_time_channel, ApplyTimeChannel, \
    cir_to_time_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber, hard_decisions
import sionna.phy as sn
from src.qQ_Method.qQ_creator_layer import qQ_creator_conv_gru
from src.qQ_Method.Q_Modulator import Q_Modulator
from src.qQ_Method.Q_Demodulator import Q_Demodulator
from src.qQ_Method.qQ_uncertainty_model import UncertaintyModel_1D, UncertaintyModel_2D
from utils.PAPR import emprical_papr


class MLPEqualizer(keras.layers.Layer):
    """
    Lightweight MLP joint equalizer.

    Input:  (batch, 128) real — flattened Q-demodulated symbols (64 complex -> 128 real)
    Output: (batch, 128) real — equalized symbols, added as residual to input

    Design principles:
      - Residual connection: MLP learns a CORRECTION, not the full mapping
      - Small capacity (~130K params): fast convergence, less gradient noise
      - Processes all 64 symbols jointly (unlike per-symbol one-tap eq)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(256, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(256, use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.dense_out = keras.layers.Dense(128, use_bias=False,
                                             kernel_initializer='zeros')  # zero init -> no correction initially

    def call(self, r_flat, training=False):
        """
        Args:
            r_flat: (batch, 128) real — flattened equalized symbols
        Returns:
            (batch, 128) real — residual correction added to input
        """
        x = tf.nn.relu(self.bn1(self.dense1(r_flat), training=training))
        x = tf.nn.relu(self.bn2(self.dense2(x), training=training))
        correction = self.dense_out(x)  # zero-init -> starts as no correction
        return r_flat + correction


class qQ_MODEL_TV_CNN(keras.Model):
    """
    OTFS waveform learning with MLP joint equalizer.

    Transmit chain:  Q-creator -> Q_Modulator -> channel  (unchanged)
    Receive chain:   Q_Demodulator -> MLPEqualizer -> Demapper
    """

    def __init__(self, training=False, visualize=False, BS_ant=1, UT_ant=1):
        super().__init__()
        self.CCDF_mode = False
        self.visualize_progress = visualize

        self._tot_symbol_to_deliver = TOT_SYMBOLS_TO_DELIVER
        self._carrier_frequency = CARRIER_FREQ
        self._subcarrier_spacing = SUBCARRIER_SPACING
        self._fft_size = FFT_SIZE
        self._cyclic_prefix_length = CYCLIC_PRFX_LEN
        self._num_ofdm_symbols = NUM_OFDM_SYMBOL
        self._num_ut_ant = UT_ant
        self._num_bs_ant = BS_ant
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = False
        self._num_guard_carriers = [0, 0]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = OFDM_SYMBOLS_FOR_PILOT_INDICES
        self._num_bits_per_symbol = NUM_BITS_PER_SYMBOL
        self._coderate = 1

        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size, subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=1, num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers, dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)

        self._channel_model = channel_model
        l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_min = L_MIN if L_MIN is not None else l_min
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                              l_tot=self._l_tot, add_awgn=True)

        self._binary_source = BinarySource()
        self._mapper = Mapper("pam" if self._num_bits_per_symbol == 1 else 'qam',
                              self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self.OFDM_modulator = OFDMModulator(self._cyclic_prefix_length)
        self.OFDM_demodulator = OFDMDemodulator(self._fft_size, self._l_min,
                                                 self._cyclic_prefix_length)

        self.training = training
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

        # Transmit
        self._qQ_creator_layer = qQ_creator_conv_gru(self._fft_size)
        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)

        # Receive
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min,
                                             self._cyclic_prefix_length)
        self._mlp_equalizer = MLPEqualizer()
        self._demapper = Demapper("app",
                                  "pam" if self._num_bits_per_symbol == 1 else 'qam',
                                  self._num_bits_per_symbol, hard_out=False)

        # Uncertainty
        self._UncertaintyModel_bce_par = UncertaintyModel_2D()
        self._UncertaintyModel_par_lim = UncertaintyModel_1D()

    def call(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        a, tau = self._channel_model(batch_size,
                                     self._rg.num_time_samples + self._l_tot - 1,
                                     self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                     l_min=self._l_min, l_max=self._l_max, normalize=True)

        # Q creation
        h_2d = h_time[:, 0, 0, 0, 0, :, :]
        total_time = tf.shape(h_2d)[1]
        stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
        indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
        pilots_post_channel = tf.gather(h_2d, indices, axis=1)
        pilots_post_channel = tf.transpose(pilots_post_channel, [0, 2, 1])

        # Channel features
        h_mag = tf.abs(pilots_post_channel)
        h_first = h_mag[:, :, 0]
        delays = tf.cast(tf.range(tf.shape(h_first)[1]), tf.float32)
        power = tf.square(h_first)
        total_power = tf.reduce_sum(power, axis=-1, keepdims=True) + 1e-10
        mean_delay = tf.reduce_sum(delays[None, :] * power, axis=-1, keepdims=True) / total_power
        rms_ds = tf.sqrt(tf.reduce_sum(power * tf.square(delays[None, :] - mean_delay), axis=-1)
                        / tf.squeeze(total_power, -1))
        h_var = tf.math.reduce_variance(h_mag, axis=-1)
        doppler_ind = tf.reduce_mean(h_var, axis=-1)
        h_avg = tf.reduce_mean(h_mag, axis=-1)
        threshold_pwr = 0.1 * tf.reduce_max(h_avg, axis=-1, keepdims=True)
        n_taps = tf.reduce_sum(tf.cast(h_avg > threshold_pwr, tf.float32), axis=-1)
        channel_features = tf.stack([
            rms_ds * 1e9, doppler_ind * 1e3, n_taps,
            tf.math.log(rms_ds * 1e9 + 1e-10),
            tf.math.log(doppler_ind * 1e3 + 1e-10)
        ], axis=-1)

        Q, q = self._qQ_creator_layer(pilots_post_channel, training=self.training)
        x_time = self._Q_modulator(Q, x_rg)

        if self.CCDF_mode:
            return x_time[:, 0, 0, :], tf.expand_dims(rms_ds, -1), channel_features

        y_time = self._channel_time(x_time, h_time, no)

        # Q^H Demodulation
        r_freq = self._Q_demodulator(Q, y_time)           # (B,1,1,3,32)
        r_freq = r_freq[:, :, :, 1:, :]                   # (B,1,1,2,32) — remove pilot

        # Apply q (per-subcarrier scaling) — same as original one-tap eq
        # This is ESSENTIAL: the pre-trained Q-creator's q provides proper scaling.
        # Without it, symbols are mis-scaled and equalization fails.
        q_bc = q[:, tf.newaxis, tf.newaxis, tf.newaxis, :]  # (B,1,1,1,32)
        q_bc = tf.tile(q_bc, [1, 1, 1, tf.shape(r_freq)[-2], 1])  # (B,1,1,2,32)
        r_freq = r_freq * q_bc                              # per-subcarrier scaling

        # MLP Equalizer: flatten -> equalize -> unflatten
        r_freq_data = r_freq[:, 0, 0, :, :]               # (B, 2, 32) complex
        r_flat = tf.concat([tf.reshape(tf.math.real(r_freq_data), [batch_size, -1]),
                            tf.reshape(tf.math.imag(r_freq_data), [batch_size, -1])], axis=-1)  # (B, 128)
        r_eq_flat = self._mlp_equalizer(r_flat, training=self.training)  # (B, 128)
        r_eq_real = r_eq_flat[:, :64]
        r_eq_imag = r_eq_flat[:, 64:]
        r_eq = tf.complex(r_eq_real, r_eq_imag)            # (B, 64)
        r_eq = tf.reshape(r_eq, [-1, 1, 1, self._tot_symbol_to_deliver])
        r_eq.set_shape([None, None, None, self._tot_symbol_to_deliver])

        # Demapper
        llr = self._demapper(r_eq, no)                     # (B,1,1,256)

        # Uncertainty
        log_sigma_par, log_sigma_bce, log_sigma_par_lim = self._UncertaintyModel_bce_par(
            channel_features, training=self.training)
        par_lim = self._UncertaintyModel_par_lim(channel_features, training=self.training)

        if self.training:
            bce_loss = tf.squeeze(self.bce(tf.reshape(b, tf.shape(llr)), llr))
            if USE_PAPR_LOSS:
                PAR = emprical_papr(tf.squeeze(x_time), None, par_lim)
                total_loss = tf.reduce_mean(
                    tf.exp(-log_sigma_bce) * bce_loss + tf.exp(-log_sigma_par) * PAR
                    + tf.exp(-log_sigma_par_lim) * par_lim
                    + log_sigma_bce + log_sigma_par + log_sigma_par_lim)
            else:
                total_loss = tf.reduce_mean(bce_loss)
            return total_loss
        else:
            b_hat = hard_decisions(llr)
            b_hat = tf.reshape(b_hat, tf.shape(b))
            return b, b_hat


if __name__ == "__main__":
    sn.config.seed = SEED
    model = qQ_MODEL_TV_CNN(training=True)
    loss = model(10, 20.0)
    print("Forward pass OK, loss:", float(loss))
