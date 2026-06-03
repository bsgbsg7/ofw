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
                              empirical_aclr,RootRaisedCosineFilter
from sionna.phy.utils import ebnodb2no, compute_ber
from channel import channel_model

class SCRRC(keras.Model): # Inherits from Keras Model

    def __init__(self,visualize=False):
        super().__init__()

        # General 
        self.CCDF_mode = False
        self._visulaize = visualize
        
        # System Parameters
        self._num_bits_per_symbol = NUM_BITS_PER_SYMBOL
        self._num_symbols = NUM_SC_SYMBOL

        # SC filter params
        self._span_in_symbols = 32 # Filter span in symbold - if equal 6 The pulse-shaping filter (e.g., root-raised-cosine) will span 6 symbol durations
        self._samples_per_symbol = 3 # Number of samples per symbol, i.e., the oversampling factor
        self._roll_off_factor = 0.001
        self._rrcf = RootRaisedCosineFilter(self._span_in_symbols, self._samples_per_symbol, self._roll_off_factor)
        
        # Required system components
        self._binary_source = sn.mapping.BinarySource()
        self._constellation = sn.mapping.Constellation("pam" if self._num_bits_per_symbol == 1 else 'qam', self._num_bits_per_symbol)
        self._mapper = sn.mapping.Mapper(constellation=self._constellation)
        self._symbol2bits = sn.mapping.SymbolInds2Bits(num_bits_per_symbol=self._num_bits_per_symbol)
        self._demapper = sn.mapping.Demapper('app', constellation=self._constellation, hard_out=True)

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

    # @tf.function 
    def __call__(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, 1) 
        bits = self._binary_source([batch_size, self._num_symbols, self._num_bits_per_symbol])
        s = self._mapper(bits)
        s = tf.reshape(s,[-1,self._num_symbols])

        # transmiter
        s_upsampled = self._us(s) # up sampling
        x_rrcf = self._rrcf(s_upsampled) # shape filtering

        # CCDF Mode
        if self.CCDF_mode:
            return x_rrcf

        # Channel
        channel_out, h_time = self._channel(tf.reshape(x_rrcf, [tf.shape(x_rrcf)[0], 1, 1, -1]), no)
        channel_out = tf.squeeze(channel_out, axis=[1, 2])
        channel_out = channel_out[:,-self._channel._l_min:-self._channel._l_max]     # slice channel excess edges l_min and l_max
        # channel_out = self._awgn_channel(x_rrcf, no)

        # reciver
        x_mf = self._rrcf(channel_out)                                               # Apply the matched filter
        s_hat = self._ds(x_mf) 

        if self._visulaize:
            self.visulaize(x_recive=x_mf,
                        x_transmit=x_rrcf,
                        s=s, 
                        s_hat=s_hat, 
                        s_upsampled=s_upsampled, 
                        channel_out=channel_out, 
                        PA_out=None, 
                        h_time=h_time)

        b_hat = self._demapper(s_hat, no)
        b_hat = tf.reshape(b_hat, bits.shape)
        # compute_ber(bits, b_hat)
        return bits, b_hat

    def visulaize(self,x_recive, x_transmit, s, s_hat, s_upsampled, channel_out, PA_out, h_time,  **kwargs):

        custom_title = kwargs.get('custom_title', None)
        ACLR = empirical_aclr(x_transmit, oversampling=self._samples_per_symbol).numpy()
        CCDF = None

        # Time domain 
        plt.figure(figsize=(12, 8))
        plt.plot(np.real(s_upsampled[0]), "x")
        plt.plot(np.real(x_transmit[0, self._rrcf.length//2:]))
        plt.plot(np.real(channel_out[0, self._rrcf.length//2:]))
        plt.plot(np.real(x_recive[0, self._rrcf.length-1:]))
        plt.xlim(-10,50)
        plt.legend([r"Oversampled sequence of QAM symbols $x_{us}$",
                    r"Transmitted sequence after pulse shaping $x_{rrcf}$",
                    r"Channel output",
                    r"Received sequence after matched filtering $x_{mf}$"])
        plt.savefig('_timeDomain.png')
        
        # Constalation
        plt.figure()
        plt.scatter(np.real(s_hat), np.imag(s_hat))
        plt.scatter(np.real(s), np.imag(s))
        plt.legend(["Received", "Transmitted"]) 
        plt.title("Scatter plot of the transmitted and received QAM symbols")
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.savefig('_constalation.png')
        print("MSE between x and x_hat (dB)", 10*np.log10(np.var(s-s_hat)))
        
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

if __name__ == "__main__":
    sn.config.seed = 1
    model = SCRRC(visualize=True)
    model(128,70)