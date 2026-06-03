import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import sionna.phy as sn
from config import *
import keras
import matplotlib.pyplot as plt
import numpy as np
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
import tensorflow as tf
from sionna.phy.channel import  ApplyTimeChannel, TimeChannel
from sionna.phy.signal import Upsampling, Downsampling, \
                              SincFilter, empirical_psd, \
                              CustomFilter, \
                              empirical_aclr
from sionna.phy.utils import ebnodb2no, compute_ber, hard_decisions  
from channel import channel_model
from src.E2EWL.Gtx_Grx_filter_coeffs import Receive_coeffs, Transmit_coeffs
from src.E2EWL.NeuralReciver import E2EWLNeuralReciver
from utils.PAPR import emprical_papr

class E2EWL_MODEL(keras.Model): # Inherits from Keras Model
    ''' 
    https://ieeexplore.ieee.org/document/9747919
    '''
    def __init__(self,
                 training=False,
                 load_pretrained=False,
                 is_neural_reciver=True,
                 is_multypath=True,
                 visualize=False):
        super().__init__()

        # General
        self.CCDF_mode = False
        self.visulaize_progress = visualize
        self._is_multypath = is_multypath
        self.init_rxtx_filters_coefficiets_as_rrcf = True
        self.init_learnbale_constaltion_from_noraml_dist = False

        # System Parameters
        self._num_bits_per_symbol = NUM_BITS_PER_SYMBOL
        self._num_symbols = NUM_SC_SYMBOL                   # total number of symbols, including pilots and data
        self._Np = 32                                       # number of symbols that are pitlos the from total self._num_symbols symbols (in case _num_symbols=64 and _Np=32, half of the block is pilots)

        # SC filter params
        self._span_in_symbols = 32                          # Filter span in symbold - if equal 6 The pulse-shaping filter (e.g., root-raised-cosine) will span 6 symbol durations
        self._samples_per_symbol = 3                        # Number of samples per symbol, i.e., the oversampling factor
        
        # Leanrable Transmit and Recive filters
        self._rrcf = SincFilter(self._span_in_symbols, self._samples_per_symbol)
        self._transmit_coeff = Transmit_coeffs(shape=self._rrcf.coefficients.shape)
        self._receive_coeff = Receive_coeffs(shape=self._rrcf.coefficients.shape)
        self._transmit_filter = CustomFilter(self._samples_per_symbol, self._transmit_coeff(None), normalize=True)
        self._recive_filter = CustomFilter(self._samples_per_symbol, self._receive_coeff(None), normalize=False)
        # initlized rrcf coefficiets values to the filters
        if self.init_rxtx_filters_coefficiets_as_rrcf:
            self._transmit_filter.coefficients.assign(tf.complex(self._rrcf.coefficients, tf.zeros_like(self._rrcf.coefficients))) 
            self._recive_filter.coefficients.assign(tf.complex(self._rrcf.coefficients, tf.zeros_like(self._rrcf.coefficients))) 

        # Learnable constelation params
        initial_constellation_points = sn.mapping.Constellation("pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol,normalize=True,center=True).points
        # Initialze learnable contaletion points from normal distribution
        if self.init_learnbale_constaltion_from_noraml_dist:
            initial_constellation_points = tf.complex(tf.random.normal((tf.size(initial_constellation_points),),stddev=0.01),  # uncomment here for random inital constellation values
                                                        tf.random.normal((tf.size(initial_constellation_points),),stddev=0.01))
        self._constelation_points = tf.Variable(tf.stack([tf.math.real(initial_constellation_points),
                                                          tf.math.imag(initial_constellation_points)], axis=0),
                                                name='constelation_points',
                                                trainable=True)
        
        self._constellation = sn.mapping.Constellation("custom",
                                                    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                                    points = tf.complex(self._constelation_points[0], self._constelation_points[1]),
                                                    normalize=True,
                                                    center=True)        
        # Learnable Pilots
        if self._is_multypath:
            self._is_learnable_pilots = True            
            self._pilot_seq = np.zeros([self._Np], np.complex64)
            self._pilot_seq[0:self._Np:2] = (1+1j)/np.sqrt(2)
            self._pilot_seq[1:self._Np:2] = (-1-1j)/np.sqrt(2)  
            self._pilot_seq = tf.Variable(self._pilot_seq, name='trainable_pilots', trainable=self._is_learnable_pilots )

        # Neural Reciver params
        self._is_neural_reciver = is_neural_reciver
        self._Ns = 64 # number of pilots features
        self._NeuralReciver = E2EWLNeuralReciver(self._num_bits_per_symbol, self._Np, self._Ns, self._is_multypath)

        # Required system components
        self._binary_source = sn.mapping.BinarySource()
        self._mapper = sn.mapping.Mapper(constellation=self._constellation)
        self._symbol2bits = sn.mapping.SymbolInds2Bits(num_bits_per_symbol=self._num_bits_per_symbol)
        self._demapper = sn.mapping.Demapper('app', constellation=self._constellation, hard_out=False)

        self._us = Upsampling(self._samples_per_symbol)
        self._ds = Downsampling(self._samples_per_symbol, self._rrcf.length-1, self._num_symbols)

        # Channel
        self._channel_model = channel_model
        self._awgn_channel = sn.channel.AWGN()
        num_time_samples = self._num_symbols * self._samples_per_symbol + self._rrcf.length-1
        bandwidth = SYMBOL_RATE
        self._channel = TimeChannel(self._channel_model, bandwidth, num_time_samples,
                                    l_min=L_MIN, l_max=None, normalize_channel=True,
                                    return_channel=True)

        # Training and Loss
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.training = training
        self.load_pretrained = load_pretrained

        self._epsilon_A = -0               # [dB]
        self._epsilon_P = 8               # [dB]
        
        self._lamda_A = 0.0
        self._lamda_P = 0.0
        self._eta     = 1e-2

    # @tf.function 
    def __call__(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, 1)
        bits = self._binary_source([batch_size, self._num_symbols, self._num_bits_per_symbol])
        
        # Assign points to constellation
        self._mapper.constellation.points = tf.complex(self._constelation_points[0], self._constelation_points[1])        
        
        s = self._mapper(bits)
        s = tf.reshape(s,[-1,self._num_symbols])

        # insert pilots
        if self._is_multypath:
            mean = tf.reduce_mean(self._pilot_seq)                                                   
            energy_sqrt = tf.complex(tf.sqrt(tf.reduce_mean(tf.square(tf.abs(self._pilot_seq )))), tf.constant(0.0))            
            s = tf.concat([ tf.tile(tf.expand_dims((self._pilot_seq - mean) / energy_sqrt, axis=0), [tf.shape(s)[0], 1]),   # normlize and center
                            s[:, self._Np:]], 
                            axis=1)
            bits = bits[:, self._Np:,:]

        # transmiter
        s_upsampled = self._us(s) # up sampling
        x_transmit = self._transmit_filter(s_upsampled) # shape filtering
        
        # CCDF Mode
        if self.CCDF_mode:
            return x_transmit, None
        
        # Channel
        channel_out, h_time = self._channel(tf.reshape(x_transmit, [tf.shape(x_transmit)[0], 1, 1, -1]), no)
        channel_out = tf.squeeze(channel_out, axis=[1, 2])
        channel_out = channel_out[:,-self._channel._l_min:-self._channel._l_max]     # slice channel excess edges l_min and l_max
        # channel_out = self._awgn_channel(x_transmit, no)

        # reciver
        x_recive = self._recive_filter(channel_out)                                  # Apply the matched filter
        r_m = self._ds(x_recive) 

        if self.visulaize_progress:
            self.visulaize(x_recive=x_recive, 
                        x_transmit=x_transmit, 
                        s=s, 
                        s_hat=r_m, 
                        s_upsampled=s_upsampled, 
                        channel_out=channel_out, 
                        PA_out=None, 
                        h_time=h_time, 
                        transmit_filter=self._transmit_filter,
                        recive_filter=self._recive_filter)

        if self._is_neural_reciver:
            llr = self._NeuralReciver(r_m, training=self.training)
        else:
            llr = self._demapper(r_m, no)
        
        if self.training:   
            
            ACLR = empirical_aclr(x_transmit, oversampling=self._samples_per_symbol)
            PAR  = self.E2EWL_emprical_papr(x_transmit, self._transmit_filter.length, self._epsilon_P)
            
            bce_loss  = self.bce(tf.reshape(bits,tf.shape(llr)), llr)
            ACLR_loss = tf.math.maximum(ACLR - tf.constant(np.power(10.0, self._epsilon_A / 10.0),dtype=tf.float32), tf.constant(0.0,dtype=tf.float32))
            PAR_loss  = PAR #tf.math.maximum(PAR - tf.constant(np.power(10.0, self._epsilon_P / 10.0),dtype=tf.float32), tf.constant(0.0,dtype=tf.float32))
            
            ACLR_loss_squere = tf.math.square(ACLR_loss)
            PAR_loss_squere = tf.math.square(PAR_loss)
            
            # Total loss
            total_loss = bce_loss - self._lamda_P * PAR_loss - self._lamda_A * ACLR_loss + (0.001 / 2) * (ACLR_loss_squere + PAR_loss_squere)
            
            # updates lagrange parameters
            self._eta *= 1.003
            self._lamda_A = self._lamda_A - self._eta * ACLR_loss
            self._lamda_P = self._lamda_P -  self._eta * PAR_loss

            # log
            self.training_log(total_loss, bce_loss, ACLR, PAR, llr, bits)
            return total_loss
        
        else:
            b_hat = hard_decisions(llr)
            b_hat = tf.reshape(b_hat, bits.shape)
            # compute_ber(bits, b_hat)
            return bits, b_hat      

    def visulaize(self, x_recive, x_transmit, s, s_hat, s_upsampled, channel_out, PA_out, h_time, transmit_filter, recive_filter,  **kwargs):
        
        custom_title = kwargs.get('custom_title', None)
        ACLR = empirical_aclr(x_transmit, oversampling=self._samples_per_symbol).numpy()
        CCDF = None

        # Time domain 
        plt.figure(figsize=(12, 8))
        plt.plot(np.real(s_upsampled[0]), "x")
        plt.plot(np.real(x_transmit[0, self._rrcf.length//2:]))
        plt.plot(np.real(channel_out[0, self._rrcf.length//2:]))
        plt.plot(np.real(x_recive[0, self._rrcf.length-1:]))
        plt.xlim(0,100)
        plt.legend([r"Oversampled sequence of QAM symbols $x_{us}$",
                    r"Transmitted sequence after pulse shaping $x_{rrcf}$",
                    r"Channel output",
                    r"Received sequence after matched filtering $x_{mf}$"])
        plt.savefig('_timeDomain.png')
        plt.close()
        
        # Constalation
        plt.figure()
        # plt.scatter(np.real(s_hat), np.imag(s_hat))
        # plt.scatter(np.real(s), np.imag(s))
        plt.scatter(np.real(s[:, :self._Np]), np.imag(s[:, :self._Np]), color='red')
        plt.scatter(np.real(s[:, self._Np:]), np.imag(s[:, self._Np:]), color='blue')        
        plt.legend(["Pilots", "Learned Constaletion points", "Learned Constaletion"]) 
        plt.title("Scatter plot of the transmitted and received QAM symbols")
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.savefig('_constalation.png')
        plt.close()
        
        # PSD - transmit filter and recive filter
        transmit_filter.show(response="impulse", scale='lin') # for time domian - 'impulse' for specturm: 'magnitude'
        plt.ylim(-1,1)
        plt.savefig('_transmit_filter.png')
        plt.close()
        recive_filter.show(response="impulse", scale='lin') # for time domian - 'impulse' for specturm: 'magnitude'
        plt.ylim(-2,2)
        plt.savefig('_recive_filter.png')
        plt.close()

        # PSD - Transmited signal
        freqs_x_transmit_normalized, psd_x_transmit = empirical_psd(x_transmit, oversampling=self._samples_per_symbol, show=False)
        freqs_channel_out_normalized, psd_channel_out = empirical_psd(channel_out, oversampling=self._samples_per_symbol, show=False)
        freqs_x_transmit = freqs_x_transmit_normalized * SYMBOL_RATE
        freqs_channel_out = freqs_channel_out_normalized * SYMBOL_RATE
        # freqs, psd_PostPA = empirical_psd(PA_out, oversampling=self._samples_per_symbol, show=False)
        plt.figure()
        plt.plot(freqs_x_transmit / 1e6, 10*np.log10(psd_x_transmit), label='Channel In', color='red')
        plt.plot(freqs_channel_out / 1e6, 10*np.log10(psd_channel_out), label='Channel Out', color='blue')
        # plt.plot(freqs, 10*np.log10(psd_PostPA), label='PA Output', color='green')
        plt.title(f"Power Spectral Density - ACLR: {ACLR:.3f}")
        plt.xlabel("Frequency [MHz]")
        plt.xlim([freqs_x_transmit[0] / 1e6, freqs_x_transmit[-1] / 1e6])
        plt.ylabel(r"$\mathbb{E}\left[|X(f)|^2\right]$ (dB)")
        ylim=[-50, 3]
        plt.ylim(ylim)
        plt.legend()
        plt.grid(True, which="both")        
        plt.savefig('_transmited_signal_psd.png')
        plt.close()

        # channel Impulse response
        plt.figure()
        plt.title("Discrete-time channel impulse response")
        plt.stem(np.abs(h_time[3,0,0,0,0,0]))
        plt.xlabel(r"Time step $\ell$")
        plt.ylabel(r"$|\bar{h}|$");
        plt.savefig('_channel.png')
        plt.close()
        return {'ACLR':ACLR, 'CCDF':CCDF}

    def training_log(self, total_loss, bce_loss, ACLR, PAR, llr, bits):
        b_hat = hard_decisions(llr)
        b_hat = tf.reshape(b_hat, bits.shape)
        ber = compute_ber(bits, b_hat)
        print("Total Loss:",f'{total_loss.numpy():.5f}',
               '... BCE Loss:', f'{bce_loss.numpy():.5f}' ,
               '... ACLR:', f'{ACLR.numpy():.5f}' ,
               '... PAR', f'{PAR.numpy():.5f}',
               '... Block Error Rate:', f'{ber.numpy():.5f}')

        return {'Total_Loss': total_loss, 'bce_Loss':bce_loss.numpy(), 'BER':ber.numpy()}

    def E2EWL_emprical_papr(self, x, T, epsilon_P):
        # Compute instantaneous power
        p_t = tf.abs(x)**2
        # compute mean power
        p_bar = tf.expand_dims(tf.reduce_mean(tf.abs(x)**2,  axis=[-1]), axis=-1) 
        # Compute max
        p_t_max = tf.math.maximum((p_t / p_bar) - tf.constant(np.power(10.0, epsilon_P / 10.0),dtype=tf.float32), tf.constant(0.0,dtype=tf.float32))
        return tf.reduce_mean(p_t_max)

if __name__ == "__main__":
    sn.config.seed = 1
    model = E2EWL_MODEL(training=True,
                        is_multypath=True,
                        is_neural_reciver=True,
                        visualize=True)
    model(128,70)