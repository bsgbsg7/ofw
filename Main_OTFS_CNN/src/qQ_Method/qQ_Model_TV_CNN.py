"""
qQ Model with Time-Varying channel + CNN equalizer (Demapper retained).

Key difference from Main_OTFS/qQ_Model_TV.py:
  The one-tap equalizer (per-subcarrier r_freq * q) is replaced by a
  complex-valued CNN that does JOINT equalization across the 2x32
  time-frequency grid. The Sionna Demapper is RETAINED — it already
  knows the 16QAM constellation and computes proper LLRs.

  CNN equalizer: r_freq (2x32) → cleaned symbols (2x32)
  Demapper: cleaned symbols → LLRs → BCE loss

  For OTFS (dense Q spreading symbols across time-frequency), the CNN
  equalizer can learn the corresponding despreading operation, which
  requires joint processing — exactly what a CNN provides.
"""
import tensorflow as tf
import keras
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import *
from channel_tv import channel_model
import numpy as np
from sionna.phy import Block
from sionna.phy.mimo.stream_management import StreamManagement
from sionna.phy.ofdm.resource_grid import ResourceGrid
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.channel import time_lag_discrete_time_channel, OFDMChannel, ApplyTimeChannel, \
    cir_to_time_channel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber, flatten_last_dims, hard_decisions
import sionna.phy as sn
from src.qQ_Method.qQ_creator_layer import qQ_creator_conv_gru
from src.qQ_Method.Q_Modulator import Q_Modulator
from src.qQ_Method.Q_Demodulator import Q_Demodulator
from src.qQ_Method.qQ_uncertainty_model import UncertaintyModel_1D, UncertaintyModel_2D
from utils.PAPR import emprical_papr
import matplotlib.pyplot as plt


class CNNEqualizer(keras.layers.Layer):
    """
    Complex-valued CNN equalizer for joint time-frequency processing.

    Input:  (batch, 2, 32) complex — Q-demodulated 2x32 time-freq grid
    Output: (batch, 2, 32) complex — equalized symbols (cleaned for demapper)

    Uses real-valued convolutions on stacked real/imag parts, applies a
    residual connection so the CNN learns a CORRECTION to the input symbols
    rather than having to reconstruct them from scratch.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        reg = keras.regularizers.l2(1e-4)

        # Conv2D on (B, 2, 32, 2) real-valued grid [time, freq, real/imag]
        self.conv1 = keras.layers.Conv2D(32, (2, 3), padding='same', use_bias=False,
                                          kernel_regularizer=reg)
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv2D(32, (2, 3), padding='same', use_bias=False,
                                          kernel_regularizer=reg)
        self.bn2 = keras.layers.BatchNormalization()

        self.conv3 = keras.layers.Conv2D(32, (2, 3), padding='same', use_bias=False,
                                          kernel_regularizer=reg)
        self.bn3 = keras.layers.BatchNormalization()

        # Output: 2 channels (real, imag) to reconstruct complex symbols
        self.conv_out = keras.layers.Conv2D(2, (2, 3), padding='same', use_bias=False,
                                             kernel_regularizer=reg)

    def call(self, r_freq, training=False):
        """
        Args:
            r_freq: (batch, 2, 32) complex — Q-demodulated data symbols
        Returns:
            (batch, 2, 32) complex — equalized symbols
        """
        # Store input for residual connection
        r_input = r_freq

        # Complex → real-valued: (B, 2, 32) → (B, 2, 32, 2)
        x = tf.stack([tf.math.real(r_freq), tf.math.imag(r_freq)], axis=-1)

        # Conv blocks
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))    # (B, 2, 32, 32)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))    # (B, 2, 32, 32)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))    # (B, 2, 32, 32)

        # Output: (B, 2, 32, 2)
        x = self.conv_out(x)

        # Residual connection: CNN learns a correction
        correction = tf.complex(x[..., 0], x[..., 1])   # (B, 2, 32)
        output = r_input + correction                     # (B, 2, 32)

        return output


class qQ_MODEL_TV_CNN(keras.Model):
    """
    OTFS waveform learning with CNN equalizer + Demapper.

    Transmit chain (unchanged):
      bits -> QAM -> resource grid -> Q-creator(CIR) -> Q_Modulator -> channel

    Receive chain:
      y_time -> Q_Demodulator(Q^H) -> pilot removal -> r_freq (2x32)
             -> CNNEqualizer -> equalized symbols (2x32)
             -> reshape -> (64,) -> Sionna Demapper -> LLRs -> BCE loss
    """

    def __init__(self, training=False, visualize=False, BS_ant=1, UT_ant=1):
        super().__init__()

        self.CCDF_mode = False
        self.visualize_progress = visualize

        # System parameters
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
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)

        # Channel
        self._channel_model = channel_model
        self._awgn_channel = sn.channel.AWGN()

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

        # Training params
        self.training = training
        self.Q_as_ifft = False
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

        # ============ Transmit chain (unchanged) ============
        self._qQ_creator_layer = qQ_creator_conv_gru(self._fft_size)
        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)

        # ============ Receive chain ============
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min,
                                             self._cyclic_prefix_length)
        # CNN equalizer: joint time-freq equalization
        self._cnn_equalizer = CNNEqualizer()

        # Sionna Demapper: constellation-aware LLR computation (RETAINED)
        self._demapper = Demapper("app",
                                  "pam" if self._num_bits_per_symbol == 1 else 'qam',
                                  self._num_bits_per_symbol,
                                  hard_out=False)

        # Uncertainty networks
        self._UncertaintyModel_bce_par = UncertaintyModel_2D()
        self._UncertaintyModel_par_lim = UncertaintyModel_1D()

    def call(self, batch_size, ebno_db):
        """Forward pass."""
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        # ---- Time-varying channel ----
        a, tau = self._channel_model(batch_size,
                                     self._rg.num_time_samples + self._l_tot - 1,
                                     self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                     l_min=self._l_min, l_max=self._l_max, normalize=True)

        # ---- Q creation: multi-snapshot CIR ----
        h_2d = h_time[:, 0, 0, 0, 0, :, :]
        total_time = tf.shape(h_2d)[1]
        stride = tf.maximum(1, total_time // NUM_TIME_SNAPSHOTS)
        indices = tf.range(0, total_time, stride)[:NUM_TIME_SNAPSHOTS]
        pilots_post_channel = tf.gather(h_2d, indices, axis=1)
        pilots_post_channel = tf.transpose(pilots_post_channel, [0, 2, 1])

        # ---- Channel features ----
        h_mag = tf.abs(pilots_post_channel)
        h_first = h_mag[:, :, 0]
        delays = tf.cast(tf.range(tf.shape(h_first)[1]), tf.float32)
        power = tf.square(h_first)
        total_power = tf.reduce_sum(power, axis=-1, keepdims=True) + 1e-10
        mean_delay = tf.reduce_sum(delays[None, :] * power, axis=-1, keepdims=True) / total_power
        rms_ds = tf.sqrt(
            tf.reduce_sum(power * tf.square(delays[None, :] - mean_delay), axis=-1)
            / tf.squeeze(total_power, -1)
        )
        h_var = tf.math.reduce_variance(h_mag, axis=-1)
        doppler_ind = tf.reduce_mean(h_var, axis=-1)
        h_avg = tf.reduce_mean(h_mag, axis=-1)
        threshold_pwr = 0.1 * tf.reduce_max(h_avg, axis=-1, keepdims=True)
        n_taps = tf.reduce_sum(tf.cast(h_avg > threshold_pwr, tf.float32), axis=-1)

        channel_features = tf.stack([
            rms_ds * 1e9,
            doppler_ind * 1e3,
            n_taps,
            tf.math.log(rms_ds * 1e9 + 1e-10),
            tf.math.log(doppler_ind * 1e3 + 1e-10)
        ], axis=-1)

        # ---- Q matrix generation ----
        Q, q = self._qQ_creator_layer(pilots_post_channel, training=self.training)

        # ---- Q Modulation ----
        x_time = self._Q_modulator(Q, x_rg)

        if self.CCDF_mode:
            return x_time[:, 0, 0, :], tf.expand_dims(rms_ds, -1), channel_features

        # ---- Channel ----
        y_time = self._channel_time(x_time, h_time, no)

        # ---- Q^H Demodulation ----
        r_freq = self._Q_demodulator(Q, y_time)           # (B, 1, 1, 3, 32)

        # ---- Remove pilot OFDM symbol ----
        r_freq = r_freq[:, :, :, 1:, :]                   # (B, 1, 1, 2, 32)

        # ---- CNN Equalizer (REPLACES one-tap eq r_freq * q) ----
        r_freq_data = r_freq[:, 0, 0, :, :]               # (B, 2, 32) complex
        r_equalized = self._cnn_equalizer(r_freq_data, training=self.training)  # (B, 2, 32) complex

        # ---- Demapper (RETAINED — knows 16QAM constellation) ----
        # Reshape back to (B, 1, 1, 2, 32) → flatten to (B, 1, 1, 64)
        r_equalized = tf.reshape(r_equalized, [-1, 1, 1, 2, self._fft_size])
        r_flat = tf.reshape(r_equalized, [-1, 1, 1, self._fft_size * 2])
        r_flat.set_shape([None, None, None, self._tot_symbol_to_deliver])

        llr = self._demapper(r_flat, no)                  # (B, 1, 1, 256)

        # ---- Uncertainty Networks ----
        log_sigma_par, log_sigma_bce, log_sigma_par_lim = self._UncertaintyModel_bce_par(
            channel_features, training=self.training)
        par_lim = self._UncertaintyModel_par_lim(
            channel_features, training=self.training)

        if self.training:
            bce_loss = tf.squeeze(self.bce(tf.reshape(b, tf.shape(llr)), llr))

            if USE_PAPR_LOSS:
                PAR = emprical_papr(tf.squeeze(x_time), None, par_lim)
                total_loss = tf.reduce_mean(
                    tf.exp(-log_sigma_bce) * bce_loss
                    + tf.exp(-log_sigma_par) * PAR
                    + tf.exp(-log_sigma_par_lim) * par_lim
                    + log_sigma_bce + log_sigma_par + log_sigma_par_lim
                )
                self.training_log(total_loss=total_loss, bce_loss=tf.reduce_mean(bce_loss),
                                  PAR=tf.reduce_mean(PAR), llr=llr, bits=b)
            else:
                total_loss = tf.reduce_mean(bce_loss)
                self.training_log(total_loss=total_loss, bce_loss=tf.reduce_mean(bce_loss),
                                  PAR=tf.constant(0.0), llr=llr, bits=b)
            return total_loss
        else:
            b_hat = hard_decisions(llr)
            b_hat = tf.reshape(b_hat, tf.shape(b))
            return b, b_hat

    def training_log(self, total_loss, bce_loss, PAR, llr, bits):
        b_hat = hard_decisions(llr)
        b_hat = tf.reshape(b_hat, tf.shape(bits))
        ber = compute_ber(bits, b_hat)
        tf.print("Total Loss:", total_loss,
                 " | BCE:", bce_loss,
                 " | PAR:", PAR,
                 " | BER:", ber)
        return {'Total_Loss': total_loss, 'bce_Loss': bce_loss, 'BER': ber}


if __name__ == "__main__":
    sn.config.seed = SEED
    model = qQ_MODEL_TV_CNN(training=True)
    loss = model(10, 20.0)
    print("Forward pass OK, loss:", float(loss))
