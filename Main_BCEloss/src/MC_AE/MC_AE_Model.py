import tensorflow as tf
import keras
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
from sionna.phy.channel import  time_lag_discrete_time_channel, OFDMChannel,  ApplyTimeChannel, cir_to_time_channel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber 

from src.MC_AE.MC_AE_Encoder import MC_AE_Encoder
from src.MC_AE.MC_AE_Decoder import MC_AE_Decoder
import sionna.phy as sn
import matplotlib.pyplot as plt


class MC_AE_MODEL(keras.Model):
    '''
    https://ieeexplore.ieee.org/abstract/document/9271932
    '''
    def __init__(self,
                 BS_ant=1,
                 UT_ant=1,
                 Q=256,
                 true_channel_as_decoder_input=False,
                 training=False,
                 load_pretrained=False):
        super().__init__()

        # General
        self.CCDF_mode = False
        self._Embedded_MC_AE = False
        self._visulaize = False 

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
        self._num_guard_carriers = [0,0]
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

        # AE-MC params
        '''
        to set (N,M) N is the fft size of non zero data, M is the 2^(fft_size*NUM_BITS_PER_SYMBOL)
        so in totol (N,M) = (fft_size, 2^(fft_size*NUM_BITS_PER_SYMBOL))
        '''
        self._L = 128
        if  self._rg.num_effective_subcarriers * NUM_BITS_PER_SYMBOL > 20:
            raise ValueError("This Shitty E2E cant handle a incomming msg of more than 20 bits (one hot vector of length M is 2^(#incoming_bits) so it is impractical)")
        self._M = np.power(2,self._rg.num_effective_subcarriers * NUM_BITS_PER_SYMBOL) # the one hot length M = 2^(m) (also can be though as rate)
        self._Q = Q
        self._N = self._rg.num_effective_subcarriers                                    # number of the sub carriers
        self._m = self._rg.num_effective_subcarriers * NUM_BITS_PER_SYMBOL              # number of bits per ofdm symbol

        # data bits length
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
        self.MC_AE_Encoder = MC_AE_Encoder(self._N, self._M, self._m, self._Embedded_MC_AE, self._L) 
        self._rg_mapper = ResourceGridMapper(self._rg)

        self._modulator = OFDMModulator(self._cyclic_prefix_length)
        self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)
        
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        
        self.MC_AE_Decoder = MC_AE_Decoder(self._M, self._Q, self._m, self._num_ofdm_symbols, self._pilot_ofdm_symbol_indices)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

        self.loss = keras.losses.SparseCategoricalCrossentropy()
        self.training = training
        self.load_pretrained = load_pretrained
        self.true_channel_as_decoder_input = true_channel_as_decoder_input

    # @tf.function 
    def call(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        
        # AE Ecoder  
        x = self.MC_AE_Encoder(b)

        x_rg = self._rg_mapper(x)
        
        # time channel
        if Time_channel:

            x_time = self._modulator(x_rg)

            # CCDF Mode
            if self.CCDF_mode:
                return x_time[:,0,0,:]

            a, tau = self._channel_model(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                            l_min=self._l_min, l_max=self._l_max, normalize=True)       
            # For perfect CSI
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
            h_true = self._remove_nulled_scs(h_freq)

            y_time = self._channel_time(x_time, h_time, no)
            y_time = y_time[...,0:-self._l_max]           # posible correction #1 for the issue below
            # y_time = self._awgn_channel(x_time, no)
            
            # reciver
            y = self._demodulator(y_time)

            # cut packet sides due to channel convolution excess length that is added at sized, and is considered to be ofdm symbol          
            # https://github.com/NVlabs/sionna/issues/958
            # y = y[:,:,:,0:-Number_of_mistaking_ofdmsymbols,:] - posible correction #2 for the issue
            if x_rg.shape != y.shape:
                raise ValueError(f"Shape mismatch: x_rg has shape {x_rg.shape}, but y has shape {y.shape} - This is due to time channel residual L_max/L_min")              

        # freq channel
        else:
            y, h_true = self._channel_freq(x_rg, no)
        
        # detector
        if self.training:
            # true channel when training
            decoder_channel_input = h_true
            # AE Decoder
            b_hat = self.MC_AE_Decoder(y, decoder_channel_input)
        else:
            # estimated channel when evaluating
            h_hat, _ = self._ls_est(y, no)
            decoder_channel_input = h_true if self.true_channel_as_decoder_input else h_hat

            # AE Decoder    
            b_hat = self.MC_AE_Decoder(y, decoder_channel_input)

        if self._visulaize:
            h_hat, _ = self._ls_est(y, no)
            self.visulaize(h_hat, h_true)

        # output generation
        if self.training:
        # output loss when training
            b = tf.reshape(b, tf.concat([tf.shape(b_hat)[:-1],[-1]], axis=0))
            b_indices = self._bits_to_decimal(b,self._m)

            b_true     = tf.reshape(b_indices, [-1])
            b_hat      = tf.reshape(b_hat, [-1,self._M])
           
            loss = self.loss(b_true, b_hat)
            return loss
        else:
        # output argmax when evaluating
            b_hat = tf.argmax(b_hat, axis=-1)
            b_hat = self._decimal_to_bits(b_hat, self._m)
            b_hat = tf.reshape(b_hat, tf.concat([b_hat.shape[:-2], [b_hat.shape[-2] * b_hat.shape[-1]]], axis=0))
            # compute_ber(b,b_hat)
            return b, b_hat
    
    def _decimal_to_bits(self,indices, num_bits):
        # indices: tensor of shape [...], dtype=int32 or int64
        indices = tf.cast(indices, tf.int32)
        
        # Create a bitmask: [num_bits - 1, ..., 0]
        bit_shifts = tf.range(num_bits - 1, -1, -1, dtype=tf.int32)  # e.g. [2,1,0] for m=3

        # Expand dims to [..., 1] so broadcasting works with bit_shifts
        expanded = tf.expand_dims(indices, axis=-1)  # shape: [..., 1]

        # Shift and mask
        bits = tf.bitwise.right_shift(expanded, bit_shifts) & 1  # shape: [..., num_bits]
        bits = tf.cast(bits, tf.float32)
        return bits

    def _bits_to_decimal(self, tensor, m):
        
        # Create weights for binary conversion: [2^(m-1), ..., 2^0]
        weights = 2 ** tf.range(m - 1, -1, -1, dtype=tensor.dtype)  # shape: [m]
        
        # Multiply bits by weights and sum
        decimal = tf.reduce_sum(tensor * weights, axis=-1)  # shape: [..., groups]
        decimal = tf.cast(decimal, tf.int32)
        return decimal

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

if __name__ == "__main__":
    # sn.config.seed = SEED
    model = MC_AE_MODEL(training=True)
    model(128,15)
