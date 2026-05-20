"""
qQ Model with Time-Varying channel input for OTFS-like waveform learning.
Key difference from Main/qQ_Model.py: feeds multiple CIR snapshots (across OFDM symbols)
to the Q-creator network so it can observe channel time-variation (Doppler).
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
from sionna.phy.channel import time_lag_discrete_time_channel, OFDMChannel, ApplyTimeChannel, cir_to_time_channel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber, flatten_last_dims, hard_decisions
import sionna.phy as sn
from src.qQ_Method.qQ_creator_layer import qQ_creator_conv_gru
from src.qQ_Method.Q_Modulator import Q_Modulator
from src.qQ_Method.Q_Demodulator import Q_Demodulator
from src.qQ_Method.qQ_uncertainty_model import UncertaintyModel_1D, UncertaintyModel_2D
from utils.PAPR import emprical_papr
import matplotlib.pyplot as plt


class qQ_MODEL_TV(keras.Model):

    def __init__(self, training=False, visulaize=False, BS_ant=1, UT_ant=1):
        super().__init__()

        self.CCDF_mode = False
        self.visulaize_progress = visulaize

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
        self._mapper = Mapper("pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        self.OFDM_modulator = OFDMModulator(self._cyclic_prefix_length)
        self.OFDM_demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol, hard_out=False)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

        # Training params
        self.training = training
        self.Q_as_ifft = False
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        self._epsilon_P = 7

        # Layers - same architecture, but input channels = num_ofdm_symbols (multi-snapshot CIR)
        self._qQ_creator_layer = qQ_creator_conv_gru(self._fft_size)
        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)
        self._UncertaintyModel_bce_par = UncertaintyModel_2D()
        self._UncertaintyModel_par_lim = UncertaintyModel_1D()

    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        # Time channel creation (now with Doppler — channel varies across time)
        a, tau = self._channel_model(batch_size, self._rg.num_time_samples + self._l_tot - 1, self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                     l_min=self._l_min, l_max=self._l_max, normalize=True)

        ## Q creation — extract MULTI-SNAPSHOT CIR to capture time-variation ##
        # Send delta pulse and observe CIR at each OFDM symbol position
        x_time_for_csi = x_rg[:, :, :, 0, :]
        delta = tf.tile(
            tf.reshape(tf.complex(tf.one_hot(0, self._fft_size, dtype=tf.float32),
                                  tf.zeros(self._fft_size, dtype=tf.float32)),
                       [1, 1, 1, self._fft_size]),
            [tf.shape(x_rg)[0], 1, 1, 1]
        )
        diff = tf.shape(self.OFDM_modulator(x_rg))[-1] - tf.shape(x_time_for_csi)[-1]
        paddings = tf.stack([[0, 0], [0, 0], [0, 0], tf.stack([0, diff])])
        delta_padded = tf.pad(delta, paddings, mode='CONSTANT', constant_values=tf.complex(0.0, 0.0))
        y_time_csi = self._channel_time(delta_padded, h_time, 0)
        y_time_csi = y_time_csi[..., -self._l_min:-self._l_max]

        # Extract CIR at multiple time positions (each OFDM symbol boundary)
        # This captures how the channel evolves over time → Doppler information
        symbol_len = self._fft_size + self._cyclic_prefix_length
        num_snapshots = self._num_ofdm_symbols
        cir_snapshots = []
        for sym_idx in range(num_snapshots):
            start = sym_idx * symbol_len + self._cyclic_prefix_length
            snapshot = y_time_csi[:, 0, 0, start:start + self._l_max]
            cir_snapshots.append(snapshot)

        # Stack: (batch, l_max, num_snapshots) — 2D delay-time representation
        pilots_post_channel = tf.stack(cir_snapshots, axis=-1)

        # Compute a "Doppler spread" metric for uncertainty networks
        # Variance of CIR across time snapshots indicates Doppler
        h_power = tf.square(tf.abs(pilots_post_channel))
        # Time variance per delay tap
        time_var = tf.math.reduce_variance(h_power, axis=-1)
        doppler_metric = tf.reduce_mean(time_var, axis=-1, keepdims=True)
        # Also compute delay spread (RMS)
        h_avg = tf.reduce_mean(pilots_post_channel, axis=-1)
        delays = tf.range(tf.shape(h_avg)[1], dtype=tf.float32)
        power = tf.square(tf.abs(h_avg))
        mean_delay = tf.reduce_sum(delays * power, axis=-1) / (tf.reduce_sum(power, axis=-1) + 1e-10)
        rms_ds = tf.sqrt(
            tf.reduce_sum(power * tf.square(delays - mean_delay[:, None]), axis=-1)
            / (tf.reduce_sum(power, axis=-1) + 1e-10)
        )
        # Combined feature: delay spread + doppler metric
        channel_feature = tf.concat([tf.expand_dims(rms_ds, -1), doppler_metric], axis=-1)

        Q, q = self._qQ_creator_layer(pilots_post_channel, training=self.training)

        # Q Modulation
        x_time = self._Q_modulator(Q, x_rg)

        # CCDF Mode
        if self.CCDF_mode:
            return x_time[:, 0, 0, :], channel_feature

        # Channel (with Doppler)
        y_time = self._channel_time(x_time, h_time, no)
        y_time = y_time[..., -self._l_min:-self._l_max]

        # Q Demodulation
        r_freq = self._Q_demodulator(Q, y_time)

        # Decoder
        r_freq = r_freq[:, :, :, 1:, :]
        q = q[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        q = tf.tile(q, [1, 1, 1, tf.shape(r_freq)[-2], 1])
        r_freq_equalzied = r_freq * q
        current_shape = tf.shape(r_freq_equalzied)
        r_freq_equalzied = tf.reshape(r_freq_equalzied, [current_shape[0], current_shape[1], current_shape[2], -1])
        r_freq_equalzied.set_shape([None, None, None, self._tot_symbol_to_deliver])
        llr = self._demapper(r_freq_equalzied, no)

        # Uncertainty Networks — use rms_ds as feature (same interface)
        rms_ds_input = tf.expand_dims(rms_ds, -1)
        log_sigma_par, log_sigma_bce, log_sigma_par_lim = self._UncertaintyModel_bce_par(rms_ds_input, training=self.training)
        par_lim = self._UncertaintyModel_par_lim(rms_ds_input, training=self.training)

        if self.training:
            bce_loss = tf.squeeze(self.bce(tf.reshape(b, tf.shape(llr)), llr))
            PAR = emprical_papr(tf.squeeze(x_time), None, par_lim)
            total_loss = tf.reduce_mean(
                tf.exp(-log_sigma_bce) * bce_loss + tf.exp(-log_sigma_par) * PAR +
                tf.exp(-log_sigma_par_lim) * par_lim + log_sigma_bce + log_sigma_par + log_sigma_par_lim
            )
            self.training_log(total_loss=total_loss, bce_loss=tf.reduce_mean(bce_loss),
                              PAR=tf.reduce_mean(PAR), llr=llr, bits=b)
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
    model = qQ_MODEL_TV(training=True)
    model(10, 20.0)
    print("Forward pass OK")
