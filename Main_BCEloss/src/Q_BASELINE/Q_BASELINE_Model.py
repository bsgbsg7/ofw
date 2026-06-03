import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import *
from channel import channel_model
import numpy as np
from sionna.phy import Block
from sionna.phy.mimo.stream_management import StreamManagement
from sionna.phy.ofdm.resource_grid import ResourceGrid
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.channel import  time_lag_discrete_time_channel, OFDMChannel, ApplyTimeChannel, cir_to_time_channel,cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber, flatten_last_dims       
import sionna.phy as sn
from src.Q_BASELINE.Q_creator_layer import Q_creator_layer
from src.Q_BASELINE.Q_Modulator import Q_Modulator
from src.Q_BASELINE.Q_Demodulator import Q_Demodulator

import matplotlib.pyplot as plt

class Q_BASELINE_MODEL(Block):

    def __init__(self,
                 BS_ant=1,
                 UT_ant=1):
        super().__init__()

        # General 
        self.CCDF_mode = False
        self._visulaize = False # if True, remove tf.function speedup

        # System parameters
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

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
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

        # time channel
        l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_min = L_MIN if L_MIN is not None else l_min        
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                l_tot=self._l_tot,
                                                add_awgn=True)        
        # freq channel
        self._channel_freq = OFDMChannel(self._channel_model, self._rg, add_awgn=True, normalize_channel=True, return_channel=True)

        self._binary_source = BinarySource()
        self._mapper = Mapper("pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        self.OFDM_modulator = OFDMModulator(self._cyclic_prefix_length)
        self.OFDM_demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

        self._Q_creator_layer = Q_creator_layer(self._fft_size)
        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)
        
    @tf.function 
    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        
        # channel creation 
        a, tau = self._channel_model(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                        l_min=self._l_min, l_max=self._l_max, normalize=True) 

        ## Q creation ##        
        # generate CSI for Q model
        if True: 
            # use pilot ofdm_symbol exchange (LTF) to estimate channel for Q creation
            x_time_for_csi = self.OFDM_modulator(x_rg)
            y_time = self._channel_time(x_time_for_csi, h_time, no)
            # y_time = y_time[...,-0:-self._l_max]          
            y = self.OFDM_demodulator(y_time)
            h, err_var = self._ls_est(y, no)
            x_hat_debug, no_eff = self._lmmse_equ(y, h, err_var, no)
            b_hat_debug = self._demapper(x_hat_debug, no_eff)
            # compute_ber(b,b_hat_debug)
        if True: 
            # use true CSI to estimate channel for Q creation
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
            h_true = self._remove_nulled_scs(h_freq)        
            h_true = h_true[...,0,:]
        
        Q = self._Q_creator_layer(h)

        # Q modultion
        if False:
            x_freq = tf.signal.ifftshift(x_rg, axes=-1)
            x_time = tf.tensordot(x_freq, Q, axes=[-1, 0])
            cp = x_time[...,tf.shape(x_time)[-1]-self._cyclic_prefix_length:] # Obtain cyclic prefix
            x_time = tf.concat([cp, x_time], -1) # Prepend cyclic prefix        
            x_time = flatten_last_dims(x_time, 2)
        else:
            x_time = self._Q_modulator(Q, x_rg)

        # CCDF Mode
        if self.CCDF_mode:
            return x_time[:,0,0,:]
        
        # channel
        y_time = self._channel_time(x_time, h_time, no)
        y_time = y_time[...,-self._l_min:-self._l_max]  
        # y_time = self._awgn_channel(x_time, no)

        # Reciver
        if False:
            # Reshape input to separate OFDM symbols
            new_shape = tf.concat(
                            [tf.shape(y_time)[:-1],
                            [self._num_ofdm_symbols],
                            [self._fft_size + self._cyclic_prefix_length]], 0)
            r_time = tf.reshape(y_time, new_shape)
            # Remove cyclic prefix
            r_time = r_time[...,self._cyclic_prefix_length:]
        
        # Q demodultion
            r_freq = tf.tensordot(r_time, tf.transpose(tf.math.conj(Q)), axes=[-1, 0])
            r_freq = tf.signal.fftshift(r_freq, axes=-1)
        else:
            r_freq = self._Q_demodulator(Q, y_time)
            
        # detector
        h_hat, err_var = self._ls_est(r_freq, no)
        x_hat, no_eff = self._lmmse_equ(r_freq, h_hat, err_var, no)

        if self._visulaize:
            self.visulaize(h_hat, h_true, Q)

        # Decoder
        b_hat = self._demapper(x_hat, no_eff)
        # compute_ber(b,b_hat)
        return b, b_hat

    def visulaize(self, h_hat, h_true, Q):
        
        # visualize chanel est.
        h_hat = h_hat[0,0,0,0,0,0,:]
        h_true = h_true[0,0,0,0,0,:]
        # h_true = h_true[0,0,0,0,0,0,:]
        plt.figure()
        plt.plot(np.real(h_true))
        plt.plot(np.imag(h_true))
        plt.plot(np.real(h_hat), "--")
        plt.plot(np.imag(h_hat), "--")
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"])
        plt.title("Comparison of channel frequency responses")
        plt.savefig('_ofdm_model_csi_estimation.png')
        plt.close()

        # time visualization of the Q modulation
        N = 3 #Q.shape[0]
        fig, axes = plt.subplots(N, 1, figsize=(10, 2*N), sharex=True)
        for i in range(N):
            row = Q[i, :]   # i-th row (complex)
            # Plot real and imaginary parts separately
            axes[i].plot(np.real(row), label="Real", color="b")
            axes[i].plot(np.imag(row), label="Imag", color="r", linestyle="--")
            axes[i].set_title(f"Row {i} of IDFT Matrix (Q)")
            axes[i].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig('_Q_functions_Time_Visualization.png')
        plt.close()

        # Freq visualization of the Q modulation
        N = Q.shape[0]
        M = 2048 * N   # larger FFT size for finer spectrum resolution
        freqs = np.fft.fftfreq(M, d=1.0)  # normalized frequency bins
        plt.figure(figsize=(10, 6))
        for i in range(N):
            row = Q[i, :]
            # zero-pad to length M before FFT
            spectrum = np.fft.fft(row, n=M)
            plt.plot(freqs, np.abs(spectrum), label=f"Row {i}")

        plt.title(f"Spectrum of IDFT Matrix Rows (FFT size = {M})")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude")
        plt.xlim(-0.01, 0.01)  # normalized frequency range
        # plt.ylim(0,11.2)
        # plt.legend(ncol=2, fontsize=8)
        plt.grid(True)
        plt.savefig('_Q_functions_Freq_Visualization.png')
        plt.close()

if __name__ == "__main__":
    sn.config.seed = SEED
    model = Q_BASELINE_MODEL()
    model(128,-10)