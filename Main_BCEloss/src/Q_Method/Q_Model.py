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
from sionna.phy.channel import  time_lag_discrete_time_channel, OFDMChannel, ApplyTimeChannel, cir_to_time_channel,cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import ebnodb2no, compute_ber, flatten_last_dims, hard_decisions       
import sionna.phy as sn
from src.Q_Method.Q_creator_layer import Q_creator_layer
from src.Q_Method.Q_Modulator import Q_Modulator
from src.Q_Method.Q_Demodulator import Q_Demodulator
from utils.PAPR import emprical_papr

import matplotlib.pyplot as plt

class Q_MODEL(keras.Model):

    def __init__(self,
                 training=False,
                 visulaize=False,
                 BS_ant=1,
                 UT_ant=1):
        super().__init__()

        # General 
        self.CCDF_mode = False
        self.visulaize_progress = visulaize # if True, remove tf.function speedup

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
        self._demapper = Demapper("app", "pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol, hard_out=False)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)


        # training params
        self.training = training
        self.Q_as_ifft = False
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self._epsilon_P = 7               # [dB]        
        self._lamda_P = 0.0
        self._eta     = 1e-3
        
        # Layers
        self._Q_creator_layer = Q_creator_layer(self._fft_size, self.Q_as_ifft)

        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

    # @tf.function 
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
        # channel input as freq domian to Q_creator    
            x_time_for_csi = self.OFDM_modulator(x_rg)
            y_time = self._channel_time(x_time_for_csi, h_time, 0)
            # y_time = y_time[...,-0:-self._l_max]          
            y = self.OFDM_demodulator(y_time)
            h, err_var = self._ls_est(y, no)
            x_hat_debug, no_eff = self._lmmse_equ(y, h, err_var, no)
            llr = self._demapper(x_hat_debug, no_eff)
            b_hat_debug = hard_decisions(llr)
            # compute_ber(b,b_hat_debug)
            channel_freq_domain = h[:,0,0,0,0,0,:]
        
        if True: 
        # channel input as time domian to Q_creator     
            x_time_for_csi_ = tf.reshape(x_rg, [x_rg.shape[0],1,1,-1])
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0,  x_time_for_csi.shape[-1]- x_time_for_csi_.shape[-1]]])
            x_time_for_csi_ = tf.pad(x_time_for_csi_, paddings, mode='CONSTANT', constant_values=0)
            y_time = self._channel_time(x_time_for_csi_, h_time, 0)
            y_time = y_time[...,-self._l_min:-self._l_max]      
            pilots_post_channel = y_time[:,0,0,:self._fft_size]   # pilots_post_channel[0,:] needs to be same as x_rg[0,0,0,0,:] in delta channel
            pilots_post_channel = tf.concat([pilots_post_channel, x_time_for_csi_[:,0,0,:self._fft_size]], axis=1)
        
        Q = self._Q_creator_layer(pilots_post_channel, training=self.training)
        
        # Q Modultion
        x_time = self._Q_modulator(Q, x_rg)

        # CCDF Mode
        if self.CCDF_mode:
            return x_time[:,0,0,:]
        
        # channel
        y_time = self._channel_time(x_time, h_time, no)
        y_time = y_time[...,-self._l_min:-self._l_max]  
        # y_time = self._awgn_channel(x_time, no)

        # Q Demodultion
        r_freq = self._Q_demodulator(Q, y_time)
        
        if self.visulaize_progress:
            self.visulaize(h, Q)

        # Decoder
        if self.Q_as_ifft: 
            # using Q as IFFT, we need equlaizer to get same results as OFDM
            r_freq, no_eff = self._lmmse_equ(r_freq, h, err_var, no)
        else:
            # count on Q to equalize (only reshaping and csi symbol removal are needed)
            r_freq = r_freq[:,:,:,1:,:] # remove first symbol which is used for csi est.
            r_freq = tf.reshape(r_freq, tf.concat([tf.shape(r_freq)[:-2],[-1]], axis=0))
        llr = self._demapper(r_freq, no)

        if self.training:
           
            # bce loss
            bce_loss  = self.bce(tf.reshape(b,tf.shape(llr)), llr)
            
            # orthogonalty loss
            QH = tf.linalg.adjoint(Q)  # conjugate transpose along last two dims
            QQH = tf.matmul(Q, QH)     # shape: [BATCH, N, N]
            I = tf.eye(self._fft_size, dtype=QQH.dtype)  # [N, N]
            I = tf.expand_dims(I, axis=0)        # [1, N, N]
            I = tf.tile(I, [tf.shape(QQH)[0], 1, 1])  # [BATCH, N, N]            
            # I = I / tf.sqrt(tf.cast(self._fft_size, I.dtype))
            norm_QQH = tf.norm(tf.math.abs(QQH - I), ord='fro', axis=[-2, -1])  # shape: [BATCH]
            orthogonalty_loss = tf.cast(tf.reduce_mean(norm_QQH),dtype='float32')
            
            # PAPR loss
            PAR  = emprical_papr(tf.squeeze(x_time), None, self._epsilon_P)
            PAR_loss = self._lamda_P * PAR
            PAR_loss_sqeure = tf.square(PAR_loss)

            total_loss = 1.0 * bce_loss + 1.0 * PAR_loss + 1e-3 * PAR_loss_sqeure
            
            self._eta *= 1.001
            self._lamda_P = self._lamda_P + self._eta * PAR  


            # log
            self.training_log(total_loss=total_loss,
                            bce_loss=bce_loss,
                            orthogonalty_loss=orthogonalty_loss,
                            ACLR=tf.constant(0),
                            PAR=PAR_loss, llr=llr, bits=b)
            return total_loss
        else:
            b_hat = hard_decisions(llr)
            b_hat = tf.reshape(b_hat, b.shape)
            # compute_ber(b,b_hat)
            return b, b_hat  

    def visulaize(self, h_true, Q):
        
        # # visualize chanel est.
        # h_true = h_true[0,0,0,0,0,:]
        # # h_true = h_true[0,0,0,0,0,0,:]
        # plt.figure()
        # plt.plot(np.real(h_true))
        # plt.plot(np.imag(h_true))

        # plt.xlabel("Subcarrier index")
        # plt.ylabel("Channel frequency response")
        # plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"])
        # plt.title("Comparison of channel frequency responses")
        # plt.savefig('_ofdm_model_csi_estimation.png')
        # plt.close()

        # # time visualization of the Q modulation
        # Q = Q[0,:,:]
        # N = 3 #Q.shape[0]
        # fig, axes = plt.subplots(N, 1, figsize=(10, 2*N), sharex=True)
        # for i in range(N):
        #     row = Q[i, :]   # i-th row (complex)
        #     # Plot real and imaginary parts separately
        #     axes[i].plot(np.real(row), label="Real", color="b")
        #     axes[i].plot(np.imag(row), label="Imag", color="r", linestyle="--")
        #     axes[i].set_title(f"Row {i} of IDFT Matrix (Q)")
        #     axes[i].legend(loc="upper right")

        # plt.tight_layout()
        # plt.savefig('_Q_functions_Time_Visualization.png')
        # plt.close()

        # # Freq visualization of the Q modulation
        # N = Q.shape[0]
        # M = 2048 * N   # larger FFT size for finer spectrum resolution
        # freqs = np.fft.fftfreq(M, d=1.0)  # normalized frequency bins
        # freqs = np.fft.fftshift(freqs)
        # plt.figure(figsize=(10, 6))
        # for i in range(2):
        #     row = Q[i, :]
        #     # zero-pad to length M before FFT
        #     spectrum = np.fft.fft(row, n=M)
        #     spectrum = np.fft.fftshift(spectrum)
        #     plt.plot(freqs, 10*np.log10(np.abs(spectrum)), label=f"Row {i}")

        # plt.title(f"Spectrum of IDFT Matrix Rows (FFT size = {M})")
        # plt.xlabel("Normalized Frequency")
        # plt.ylabel("Magnitude")
        # # plt.xlim(-0.1, 0.1)  # normalized frequency range
        # # plt.ylim(0,11.2)
        # # plt.legend(ncol=2, fontsize=8)
        # plt.grid(True)
        # plt.savefig('_Q_functions_Freq_Visualization.png')
        # plt.close()
        pass
    
    def training_log(self, total_loss, bce_loss, orthogonalty_loss, ACLR, PAR, llr, bits):
        b_hat = hard_decisions(llr)
        b_hat = tf.reshape(b_hat, bits.shape)
        ber = compute_ber(bits, b_hat)
        print("Total Loss:",f'{total_loss.numpy():.3f}',
               '... BCE :', f'{bce_loss.numpy():.5f}' ,
               '... Orthogonalty:', f'{orthogonalty_loss.numpy():.5f}' ,
               '... ACLR:', f'{ACLR.numpy():.5f}' ,
               '... PAR', f'{PAR.numpy():.15f}',
               '... Block Error Rate:', f'{ber.numpy():.15f}')

        return {'Total_Loss': total_loss, 'bce_Loss':bce_loss.numpy(), 'BER':ber.numpy()}
    
if __name__ == "__main__":
    sn.config.seed = SEED
    model = Q_MODEL(training=True, visulaize=True)
    model(10,0)