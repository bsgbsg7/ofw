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
from src.qQ_Method.qQ_creator_layer import qQ_creator_layer, OrtQ_creator_layer, qQ_creator_conv_gru
from src.qQ_Method.Q_Modulator import Q_Modulator
from src.qQ_Method.Q_Demodulator import Q_Demodulator
from src.qQ_Method.qQ_uncertainty_model import UncertaintyModel_1D, UncertaintyModel_2D
from utils.PAPR import emprical_papr
from utils.General_helpers import make_shift_P
import matplotlib.pyplot as plt

class qQ_MODEL(keras.Model):

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
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        self._epsilon_P = 7

        # Layers
        # self._qQ_creator_layer = qQ_creator_layer(self._fft_size, self.Q_as_ifft)
        # self._qQ_creator_layer = qQ_creator_transformer_layer(self._fft_size)
        self._qQ_creator_layer = qQ_creator_conv_gru(self._fft_size)
        
        # self._OrtQ_creator_layer = OrtQ_creator_layer(self._fft_size, self.Q_as_ifft)
        self._Q_modulator = Q_Modulator(self._cyclic_prefix_length)
        self._Q_demodulator = Q_Demodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)
        self._UncertaintyModel_bce_par = UncertaintyModel_2D()
        self._UncertaintyModel_par_lim = UncertaintyModel_1D()

    # @tf.function 
    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        x = self._mapper(b)
        x_rg = self._rg_mapper(x)

        
        # time channel creation 
        a, tau = self._channel_model(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
        h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                        l_min=self._l_min, l_max=self._l_max, normalize=True) 
        # freq channel creation 
        a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
        a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
        h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)
        h_freq = self._remove_nulled_scs(h_freq)
        
        ## Q creation ##        
        # generate CSI for Q model
        if False:
        # channel input as OFDM freq domian to Q_creator    
            x_time_for_csi = self.OFDM_modulator(x_rg)
            y_time = self._channel_time(x_time_for_csi, h_time, 0)
            y_time = y_time[...,-0:-self._l_max]          
            y = self.OFDM_demodulator(y_time)
            h, err_var = self._ls_est(y, no)
            x_hat_debug, no_eff = self._lmmse_equ(y, h, err_var, no)
            llr = self._demapper(x_hat_debug, no_eff)
            b_hat_debug = hard_decisions(llr)
            # compute_ber(b,b_hat_debug).numpy()
            channel_freq_domain = h[:,0,0,0,0,0,:]
        
        if True: 
        # channel input as time domian to Q_creator     
            x_time_for_csi = x_rg[:,:,:,0,:]
            delta = tf.tile(tf.reshape(tf.complex(tf.one_hot(0, self._fft_size, dtype=tf.float32), tf.zeros(self._fft_size, dtype=tf.float32)), [1,1,1,self._fft_size]), [tf.shape(x_rg)[0],1,1,1])
            diff = tf.shape(self.OFDM_modulator(x_rg))[-1] - tf.shape(x_time_for_csi)[-1]
            paddings = tf.stack([[0, 0], [0, 0], [0, 0], tf.stack([0, diff])])
            delta_padded = tf.pad(delta, paddings, mode='CONSTANT', constant_values=tf.complex(0.0, 0.0))
            y_time = self._channel_time(delta_padded, h_time, 0)
            y_time = y_time[...,-self._l_min:-self._l_max]      
            pilots_post_channel = tf.expand_dims(y_time[:,0,0,:self._l_max],axis=-1)   # pilots_post_channel[0,:] needs to be same as x_rg[0,0,0,0,:] in delta channel
        # DS in rms, for ccdf plotting and uncertainty Nets.
            h = tf.squeeze(pilots_post_channel,axis=-1)   # (batch, 21)
            delays = tf.range(tf.shape(h)[1], dtype=tf.float32)
            power = tf.square(tf.abs(h))
            mean_delay = tf.reduce_sum(delays * power, axis=-1) / tf.reduce_sum(power, axis=-1)
            rms_ds = tf.sqrt(
                tf.reduce_sum(power * tf.square(delays - mean_delay[:, None]), axis=-1)
                / tf.reduce_sum(power, axis=-1)
            )
            rms_ds = tf.expand_dims(rms_ds, -1) 

        Q, q = self._qQ_creator_layer(pilots_post_channel, training=self.training)
        
        # Q Modultion
        x_time = self._Q_modulator(Q, x_rg)

        # CCDF Mode
        if self.CCDF_mode:
            return x_time[:,0,0,:], rms_ds
        
        # channel
        y_time = self._channel_time(x_time, h_time, no)
        y_time = y_time[...,-self._l_min:-self._l_max]  
        # y_time = self._awgn_channel(x_time, no)

        # Q Demodultion
        r_freq = self._Q_demodulator(Q, y_time)
        
        # Decoder
        # q equalization
        r_freq = r_freq[:,:,:,1:,:]
        # tile q
        q = q[:, tf.newaxis, tf.newaxis, tf.newaxis, :] 
        q = tf.tile(q, [1, 1, 1, tf.shape(r_freq)[-2], 1])
        r_freq_equalzied =  r_freq * q
        current_shape = tf.shape(r_freq_equalzied)
        r_freq_equalzied = tf.reshape(r_freq_equalzied, [current_shape[0], current_shape[1], current_shape[2], -1])
        r_freq_equalzied.set_shape([None, None, None, self._tot_symbol_to_deliver]) # restore static shape
        llr = self._demapper(r_freq_equalzied, no)

        # Uncertainty Networks
        # snr = tf.expand_dims(tf.fill(tf.shape( pilots_post_channel)[0], tf.cast(ebno_db, rms_ds.dtype)),axis=-1)
        # channel_features = tf.concat([tf.math.log(rms_ds + 1e-19), snr], axis=-1)
        log_sigma_par, log_sigma_bce, log_sigma_par_lim = self._UncertaintyModel_bce_par(rms_ds, training=self.training)
        par_lim = self._UncertaintyModel_par_lim(rms_ds, training=self.training)
        
        if self.training:

            # BCE loss
            bce_loss  =  tf.squeeze(self.bce(tf.reshape(b,tf.shape(llr)), llr))
            # PAPR loss
            PAR  = emprical_papr(tf.squeeze(x_time), None, par_lim)
            # Total loss
            total_loss = tf.reduce_mean(
                                        tf.exp(-log_sigma_bce) * bce_loss + tf.exp(-log_sigma_par) * PAR + tf.exp(-log_sigma_par_lim) * par_lim + log_sigma_bce + log_sigma_par + log_sigma_par_lim
                                        )
            # log
            # self.training_log(total_loss=total_loss,
            #                     bce_loss=tf.reduce_mean(bce_loss),
            #                     PAR=tf.reduce_mean(PAR), llr=llr, bits=b)
            # visualize           
            if self.visulaize_progress:
                self.visulaize(h_freq, Q, rms_ds, tf.exp(-log_sigma_bce), tf.exp(-log_sigma_par))

            return total_loss
        
        else:
            b_hat = hard_decisions(llr)
            b_hat = tf.reshape(b_hat, tf.shape(b))
            # compute_ber(b,b_hat)
            return b, b_hat  

    def visulaize(self, h_freq, Q, ds_rms=None, w_bce=None, w_papr=None):

        ds_rms_np = ds_rms.numpy() if hasattr(ds_rms, "numpy") else ds_rms

        # Get indices for min and max DS samples in the batch
        idx_min = np.argmin(ds_rms_np)
        idx_max = np.argmax(ds_rms_np)

        Q_min = Q[idx_min,:,:]
        Q_max = Q[idx_max,:,:]
        h_freq_min = h_freq[idx_min,0,0,0,0,0,:]
        h_freq_max = h_freq[idx_max,0,0,0,0,0,:]
        N = self._fft_size
        how_many_waves_to_plots = 8
        # Visualize freq chanel min DS
        plt.figure()
        plt.plot(np.abs(h_freq_min))
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.ylim(0,2)
        plt.title(f"Channel frequency responses - DelaySpread in RMS: {ds_rms_np[idx_min]}")
        plt.savefig('_ofdm_model_csi_estimation_min_DS.png')
        plt.close()

        # Visualize freq chanel max DS
        plt.figure()
        plt.plot(np.abs(h_freq_max))
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.ylim(0,2)
        plt.title(f"Channel frequency responses - DelaySpread in RMS: {ds_rms_np[idx_max]}")
        plt.savefig('_ofdm_model_csi_estimation_max_DS.png')
        plt.close()

        # Time visualization of the Q modulation min DS
        fig, axes = plt.subplots(how_many_waves_to_plots, 1, figsize=(12, 2*how_many_waves_to_plots), sharex=True)
        for i in range(how_many_waves_to_plots):
            row = Q_min[i, :]   # i-th row (complex)
            # Plot real and imaginary parts separately
            axes[i].plot(np.real(row), label="Real", color="b")
            axes[i].plot(np.imag(row), label="Imag", color="r", linestyle="--")
            axes[i].set_title(rf"Row {i} of Q Matrix - $f_{i}[n]$")
            axes[i].set_ylabel(rf"$f_{i}[n]$")
            axes[i].set_xlabel(rf"n")
            axes[i].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('_qQ_functions_Time_Visualization_min_DS.png')
        plt.close()

        # Time visualization of the Q modulation max DS
        fig, axes = plt.subplots(how_many_waves_to_plots, 1, figsize=(12, 2*how_many_waves_to_plots), sharex=True)
        for i in range(how_many_waves_to_plots):
            row = Q_max[i, :]   # i-th row (complex)
            # Plot real and imaginary parts separately
            axes[i].plot(np.real(row), label="Real", color="b")
            axes[i].plot(np.imag(row), label="Imag", color="r", linestyle="--")
            axes[i].set_title(rf"Row {i} of Q Matrix - $f_{i}[n]$")
            axes[i].set_ylabel(rf"$f_{i}[n]$")
            axes[i].set_xlabel(rf"n")
            axes[i].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('_qQ_functions_Time_Visualization_max_DS.png')
        plt.close()

        # Freq visualization of the Q modulation min DS
        M = 2048 * N   # larger FFT size for finer spectrum resolution
        freqs = np.fft.fftfreq(M, d=1.0)  # normalized frequency bins
        freqs = np.fft.fftshift(freqs)
        plt.figure(figsize=(10, 6))
        for i in range(how_many_waves_to_plots):
            row = Q_min[i, :]
            # zero-pad to length M before FFT
            spectrum = np.fft.fft(row, n=M)
            spectrum = np.fft.fftshift(spectrum)
            # plt.plot(freqs, 10*np.log10(np.abs(spectrum)), label=f"Row {i}")
            plt.plot(freqs, np.abs(spectrum), label=f"Row {i}")

        # plt.title(f"Spectrum of IDFT Matrix Rows (FFT size = {M})")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude (linear)")
        # plt.xlim(-0.1, 0.1)  # normalized frequency range
        # plt.ylim(0,11.2)
        plt.ylim(bottom=0)
        # plt.legend(ncol=2, fontsize=8)
        plt.grid(True)
        plt.savefig('_qQ_functions_Freq_Visualization_min_DS.png')
        plt.close()

        # Freq visualization of the Q modulation max DS
        M = 2048 * N   # larger FFT size for finer spectrum resolution
        freqs = np.fft.fftfreq(M, d=1.0)  # normalized frequency bins
        freqs = np.fft.fftshift(freqs)
        plt.figure(figsize=(10, 6))
        for i in range(how_many_waves_to_plots):
            row = Q_max[i, :]
            # zero-pad to length M before FFT
            spectrum = np.fft.fft(row, n=M)
            spectrum = np.fft.fftshift(spectrum)
            # plt.plot(freqs, 10*np.log10(np.abs(spectrum)), label=f"Row {i}")
            plt.plot(freqs, np.abs(spectrum), label=f"Row {i}")

        # plt.title(f"Spectrum of IDFT Matrix Rows (FFT size = {M})")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Magnitude (linear)")
        # plt.xlim(-0.1, 0.1)  # normalized frequency range
        # plt.ylim(0,11.2)
        plt.ylim(bottom=0)
        # plt.legend(ncol=2, fontsize=8)
        plt.grid(True)
        plt.savefig('_qQ_functions_Freq_Visualization_max_DS.png')
        plt.close()

        # DS RMS vs BCE/PAPR weights
        if ds_rms is not None and w_bce is not None and w_papr is not None:
            # Convert tensors to numpy if needed
            ds_rms_np = ds_rms.numpy() if hasattr(ds_rms, "numpy") else ds_rms
            w_bce_np = w_bce.numpy() if hasattr(w_bce, "numpy") else w_bce
            w_papr_np = w_papr.numpy() if hasattr(w_papr, "numpy") else w_papr

            plt.figure(figsize=(8,6))
            plt.scatter(ds_rms_np, w_bce_np, color='b', label='BCE weight', alpha=0.7)
            plt.xlabel("RMS Delay Spread")
            plt.ylabel("BCE Loss weight")
            plt.title("Per-sample Uncertainty BCE Weights vs RMS Delay Spread")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('_ds_vs_BCE_weights.png')
            plt.close()        

            plt.figure(figsize=(8,6))
            plt.scatter(ds_rms_np, w_papr_np, color='r', label='PAPR weight', alpha=0.7)
            plt.xlabel("RMS Delay Spread")
            plt.ylabel("PAPR Loss weight")
            plt.title("Per-sample Uncertainty PAPR Weight vs RMS Delay Spread")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('_ds_vs_PAPR_weights.png')
            plt.close()   

        pass
    
    def training_log(self, total_loss, bce_loss, PAR, llr, bits, orthogonalty_loss=tf.constant([0]),):
        b_hat = hard_decisions(llr)
        b_hat = tf.reshape(b_hat, tf.shape(bits))
        ber = compute_ber(bits, b_hat)
        tf.print("Total Loss:", total_loss, 
                    " | BCE:", bce_loss, 
                    " | PAR:", PAR, 
                    " | BER:", ber)

        return {'Total_Loss': total_loss, 'bce_Loss':bce_loss, 'BER':ber}
    
if __name__ == "__main__":
    sn.config.seed = SEED
    model = qQ_MODEL(training=True, visulaize=True)
    model(100,40)
    # model(10,0)