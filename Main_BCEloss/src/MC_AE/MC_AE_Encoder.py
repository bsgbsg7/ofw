from keras import layers
from keras.layers import Dense, Layer
import tensorflow as tf
import math

class MC_AE_Encoder(Layer): # Inherits from Keras Layer

    def __init__(self,fft_size, M, m, Embedded_MC_AE, L):
        r'''
            Tot_msg_bits - total number of input bits
            m - number of input bits per ofdm symbol
            M - 2^m
            N - FFT size

            m input bits [1,...,m,...Tot_msg_bits] -> [fft_size,]
            
            first m bits one_hot([1,...,m]) of length M
            -> [1,...,Tot_msg_bits/m] -> 
            
            Input
            -----
            : [..., m], `tf.float` or `tf.int`
                Tensor with with binary entries

            Output
            ------
            : [...,M,N], `tf.complex`
                Mapped constellation symbols        
        '''
        super().__init__()

        self._Embedded_MC_AE = Embedded_MC_AE
        self._N = fft_size
        self._sqrt_N = math.sqrt(self._N)
        self._M = M
        self._L = L
        self._m = m

        if self._Embedded_MC_AE:
            self.embedding_layer = Dense(self._L, activation=None)
        self.dense = Dense(2*self._N)

    def call(self, s):
        # reshape from [...,number_of_bits] to [...,number_of_ofdm_symbols,m]
        new_shape = tf.concat([tf.shape(s)[:-1], [-1, self._m]], axis=0)
        s = tf.reshape(s, new_shape)

        if self._Embedded_MC_AE:
            # embeddings
            embedding_out = self.embedding_layer(s)
        else:
            # one hot
            indices = self._bits_to_decimal(s, self._m)
            embedding_out = tf.one_hot(indices, self._M, axis=-1) # shape: [...,Num Of OFDM Symbols,M]
            
        # dense layer
        dense_out = self.dense(embedding_out)
        
        # normalization
        normlization_out,_ = tf.linalg.normalize(dense_out, ord=2, axis=-1, name=None)
        normlization_out *= self._sqrt_N
        
        # reshapeing fron 2N to complex N 
        reshaped_out = self._convert_last_dim_to_complex(normlization_out)
        
        # reshaping along the ofdm symbols axis [Batch, 1, 1, NumOfdmSymbols, NumOfSubcarriers] -> [Batch, 1, 1, NumOfdmSymbols * NumOfSubcarriers]
        current_shape = tf.shape(reshaped_out)
        Out_shape = tf.concat([current_shape[:-2], [current_shape[-2]*current_shape[-1]]], axis=0)
        Encoder_out = tf.reshape(reshaped_out, Out_shape)
        
        return Encoder_out
    
    def _bits_to_decimal(self, tensor, m):
        
        # Create weights for binary conversion: [2^(m-1), ..., 2^0]
        weights = 2 ** tf.range(m - 1, -1, -1, dtype=tensor.dtype)  # shape: [m]
        
        # Multiply bits by weights and sum
        decimal = tf.reduce_sum(tensor * weights, axis=-1)  # shape: [..., groups]
        decimal = tf.cast(decimal, tf.int32)
        return decimal
    
    def _convert_last_dim_to_complex(self,x):
        # x: [..., 2 * N], float32 or float64
        shape = tf.shape(x)
        last_dim = tf.shape(x)[-1]

        # Reshape: [..., new_last_dim, 2]
        reshaped = tf.reshape(x, tf.concat([shape[:-1], [last_dim / 2, 2]], axis=0))

        # Split real and imaginary parts
        real = reshaped[..., 0]
        imag = reshaped[..., 1]

        # Convert to complex
        return tf.complex(real, imag)