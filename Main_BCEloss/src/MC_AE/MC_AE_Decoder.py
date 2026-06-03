from keras import layers
from keras.layers import Dense, Layer
from config import NUM_BITS_PER_SYMBOL
import tensorflow as tf

class MC_AE_Decoder(Layer): # Inherits from Keras Layer

    def __init__(self, M, Q, m, num_ofdm_symbols, pilot_symbol_indices):
        super().__init__()

        self._num_ofdm_symbols = num_ofdm_symbols
        # self._num_pilot_ofdm_symbol_indices = len(pilot_symbol_indices)
        self._num_data_symbols = num_ofdm_symbols - len(pilot_symbol_indices)
        self._M = M
        self._Q = Q
        self._m = m
        self.dense1 = Dense(self._Q, 'relu')
        self.dense2 = Dense(self._M)
    
    def call(self, y, h):       
        # remove pilot symbols which is located at the first symbol (TODO remove it according to the _pilot_ofdm_symbol_indices)
        y = y[...,1:,:]
        y_IQ = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1) # split last dim to real and image
        h_IQ = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
        h_IQ = h_IQ[..., 1:, :] # removing the first element of the second dim from the end (assuming its the pilot data)
        h_IQ = tf.reshape(h_IQ, tf.shape(y_IQ))
        nn_input =  tf.concat([y_IQ,h_IQ], axis=-1)
        
        z = self.dense1(nn_input)
        r = self.dense2(z)
        out = tf.nn.softmax(r)
        return out
    
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