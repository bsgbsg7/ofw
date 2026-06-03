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
from sionna.phy.utils import ebnodb2no, compute_ber, expand_to_rank
from sionna.phy.signal import ifft, fft
from sionna.phy.signal import Upsampling, Downsampling, SincFilter, RootRaisedCosineFilter

import sionna.phy as sn
import matplotlib.pyplot as plt

from src.SCFDE.SCFDE_modulator import SCFDEModulator
from src.SCFDE.SCFDE_demodulator import SCFDEDemodulator


class SCFDE_MODEL(Block):

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

        # Shape filter params
        self._span_in_symbols = 8 # Filter span in symbold - if equal 6 The pulse-shaping filter (e.g., root-raised-cosine) will span 6 symbol durations
        self._samples_per_symbol = 6 # Number of samples per symbol, i.e., the oversampling factor
        self._roll_off_factor = 0.15
        self._rrcf = RootRaisedCosineFilter(self._span_in_symbols, self._samples_per_symbol, self._roll_off_factor)
        self._us = Upsampling(self._samples_per_symbol)
        self._ds = Downsampling(self._samples_per_symbol, self._rrcf.length-1, self._rg.num_time_samples)        

        # Channel
        self._channel_model = channel_model
        self._awgn_channel = sn.channel.AWGN()

        # time channel
        l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_min = L_MIN if L_MIN is not None else l_min
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_input_num_time_samples = self._rg.num_time_samples * self._samples_per_symbol + self._rrcf.length-1
        self._channel_time = ApplyTimeChannel(self._channel_input_num_time_samples,
                                                l_tot=self._l_tot,
                                                add_awgn=True)        
        # freq channel
        self._channel_freq = OFDMChannel(self._channel_model, self._rg, add_awgn=True, normalize_channel=True, return_channel=True)

        self._binary_source = BinarySource()
        self._mapper = Mapper("pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        self._modulator = SCFDEModulator(self._cyclic_prefix_length)
        self._demodulator = SCFDEDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)


        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    # @tf.function 
    def call(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)
        
        # time channel
        if Time_channel:

            x_time_befor_mf = self._modulator(x_rg)
            # Up sampling and RRC filtering
            s_upsampled = self._us(x_time_befor_mf) # up sampling
            x_time = self._rrcf(s_upsampled) # shape filtering

            # CCDF Mode
            if self.CCDF_mode:
                return x_time[:,0,0,:], None


            a, tau = self._channel_model(batch_size, self._channel_input_num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                            l_min=self._l_min, l_max=self._l_max, normalize=True)       
            # For perfect CSI
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
            h_true = self._remove_nulled_scs(h_freq)

            y_time = self._channel_time(x_time, h_time, no)
            y_time = y_time[...,0:-self._l_max] # correction #1 for the issue below
            # y_time = self._awgn_channel(x_time, no)

            # Down sampling and Match filtering
            x_mf = self._rrcf(y_time)                                               # Apply the matched filter
            y_time_after_match_filter = self._ds(x_mf) 
            
            # reciver
            y = self._demodulator(y_time_after_match_filter)
            
            # cut packet sides due to channel convolution excess length that is added at sized, and is considered to be ofdm symbol          
            # https://github.com/NVlabs/sionna/issues/958
            # y = y[:,:,:,0:-Number_of_mistaking_ofdmsymbols,:] - correction #2 for the issue below
            if x_rg.shape != y.shape:
                raise ValueError(f"Shape mismatch: x_rg has shape {x_rg.shape}, but y has shape {y.shape} - This is due to time channel residual L_max/L_min")              
            
        # freq channel    
        else:
            y, h_true = self._channel_freq(x_rg, no)
        
        self._rg.pilot_pattern.pilots =  fft(self._rg.pilot_pattern.pilots) # SC/FDE requiers freq domain pilots
        # equalization
        h_hat, err_var = self._ls_est(y, no)
        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)


        # Compute IFFT 
        shape = tf.shape(x_rg)
        new_shape = tf.concat([shape[:-2], [shape[-2] - 1], shape[-1:]], axis=0)
        x_hat = tf.reshape(x_hat, new_shape)
        x_time = ifft(x_hat)
        x_time = tf.reshape(x_time, tf.shape(x))

        if self._visulaize:
            self.visulaize(h_hat, h_true)
            
        # Decoder
        b_hat = self._demapper(x_time, no_eff)
        # compute_ber(b,b_hat)
        return b, b_hat

    def visulaize(self, h_hat, h_true):
        # visualize chanel est.
        h_hat = h_hat[0,0,0,0,0,1,:]
        h_true = h_true[0,0,0,0,0,1,:]
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
        
if __name__ == "__main__":
    sn.config.seed = SEED
    model = SCFDE_MODEL()
    model(128,70)