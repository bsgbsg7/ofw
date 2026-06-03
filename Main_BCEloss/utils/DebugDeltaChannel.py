#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Class for simulating Rayleigh block fading"""

import tensorflow as tf
from sionna.phy import config
from sionna.phy.channel import ChannelModel

class DebugDeltaChannel(ChannelModel):
    # pylint: disable=line-too-long
    r"""
    Generates channel impulse responses corresponding to a Rayleigh block
    fading channel model

    The channel impulse responses generated are formed of a single path with
    zero delay and a normally distributed fading coefficient.
    All time steps of a batch example share the same channel coefficient
    (block fading).

    This class can be used in conjunction with the classes that simulate the
    channel response in time or frequency domain, i.e.,
    :class:`~sionna.phy.channel.OFDMChannel`,
    :class:`~sionna.phy.channel.TimeChannel`,
    :class:`~sionna.phy.channel.GenerateOFDMChannel`,
    :class:`~sionna.phy.channel.ApplyOFDMChannel`,
    :class:`~sionna.phy.channel.GenerateTimeChannel`,
    :class:`~sionna.phy.channel.ApplyTimeChannel`.

    Parameters
    ----------
    num_rx : `int`
        Number of receivers (:math:`N_R`)

    num_rx_ant : `int`
        Number of antennas per receiver (:math:`N_{RA}`)

    num_tx : `int`
        Number of transmitters (:math:`N_T`)

    num_tx_ant : `int`
        Number of antennas per transmitter (:math:`N_{TA}`)

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    batch_size : `int`
        Batch size

    num_time_steps : `int`
        Number of time steps

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths = 1, num_time_steps], `tf.complex`
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths = 1], `tf.float`
        Path delays [s]
    """
    def __init__(self,
                 num_rx,
                 num_rx_ant,
                 num_tx,
                 num_tx_ant,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant

    def __call__(self,  batch_size, num_time_steps, sampling_frequency=None):

        # Delays
        # Single path with zero delay
        delays = tf.zeros([ batch_size,
                            self.num_rx,
                            self.num_tx,
                            1], # Single path
                            dtype=self.rdtype)

        # Fading coefficients
        std = tf.cast(tf.sqrt(0.5), dtype=self.rdtype)
        h_real = config.tf_rng.normal(shape=[batch_size,
                                             self.num_rx,
                                             self.num_rx_ant,
                                             self.num_tx,
                                             self.num_tx_ant,
                                             1, # One path
                                             1], # Same response over the block
                                      stddev=std,
                                      dtype = self.rdtype)
        h_img = config.tf_rng.normal(shape=[batch_size,
                                            self.num_rx,
                                            self.num_rx_ant,
                                            self.num_tx,
                                            self.num_tx_ant,
                                            1, # One cluster
                                            1], # Same response over the block
                                     stddev=std,
                                     dtype = self.rdtype)
        h = tf.complex(tf.ones_like(h_real), tf.zeros_like(h_img))
        # Tile the response over the block
        h = tf.tile(h, [1, 1, 1, 1, 1, 1, num_time_steps])
        return h, delays

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from sionna.phy.channel import  TimeChannel
    import matplotlib.pyplot as plt
    import numpy as np
    from sionna.phy.channel import  time_lag_discrete_time_channel
    from channel import channel_model

    # channel_model_to_use = DebugDeltaChannel(1, 1, 1, 1)
    channel_model_to_use = channel_model

    num_time_samples = int(10000)  # Number of time samples input
    bandwidth = 5e6
    l_min, l_max = time_lag_discrete_time_channel(bandwidth)
    # l_min = 0
    timechannel = TimeChannel(channel_model_to_use, bandwidth, num_time_samples,
                                    l_min=l_min, l_max=l_max, normalize_channel=True,
                                    add_awgn=False, return_channel=True)
    
    # Ones input
    # channel_in = tf.ones([1, 1, 1, num_time_samples], dtype=tf.complex64)
    
    # Delta input
    channel_in = tf.scatter_nd(indices=[[0, 0, 0, 0]] , updates=tf.constant([1.0 + 0j], dtype=tf.complex64), shape=(1, 1, 1, num_time_samples)) 

    channel_out, h_time = timechannel(tf.reshape(channel_in, [tf.shape(channel_in)[0], 1, 1, -1]), 0.0)
    

    # visualize the channel input and output
    squeeze_channel_in =  tf.squeeze(channel_in, axis=[1, 2])
    channel_channel_out = tf.squeeze(channel_out, axis=[1, 2])
    added_length = 0
    plt.figure(figsize=(12, 8))
    plt.plot(np.abs(squeeze_channel_in[0]), "x")
    plt.plot(np.abs(channel_channel_out[0, -l_min:-l_max]))
    # plt.plot(np.real(channel_channel_out[0, :]))
    # plt.xlim(0,500)
    plt.legend([r"Channel input $x_{in}$",
                r"Chaneel output $y$"])
    plt.savefig('_Channel _IO_debug.png')

    # channel Impulse response
    plt.figure()
    plt.title("Discrete-time channel impulse response")
    plt.stem(np.abs(h_time[0,0,0,0,0,0]))
    # plt.xlim(-1,20)
    plt.xlabel(r"Time step $\ell$")
    plt.ylabel(r"$|\bar{h}|$");
    plt.savefig('_CIR_debug.png')