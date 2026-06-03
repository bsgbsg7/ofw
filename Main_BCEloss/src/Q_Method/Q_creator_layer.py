import tensorflow as tf
import numpy as np
from keras.layers import Dense, Layer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.General_helpers import idft_matrix

class Q_creator_layer(Layer):
    def __init__(self, N, return_IFFT, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(self.N)  # assume this returns complex64 matrix
        
        # Use real Dense layers for real and imag parts separately
        self.dense1_real = Dense(256, activation='relu')
        self.dense1_imag = Dense(256, activation='relu')
        self.dense2_real = Dense(256, activation='relu')
        self.dense2_imag = Dense(256, activation='relu')
        self.dense3_real = Dense(N*N)
        self.dense3_imag = Dense(N*N)

    def call(self, h, training=False):

        # Split input into real and imaginary parts
        h_real = tf.math.real(h)
        h_imag = tf.math.imag(h)
        
        # First dense layer
        x_real1 = self.dense1_real(h_real) - self.dense1_imag(h_imag)
        x_imag1 = self.dense1_real(h_imag) + self.dense1_imag(h_real)
        
        # Second dense layer
        x_real2 = self.dense2_real(x_real1) - self.dense2_imag(x_imag1)
        x_imag2 = self.dense2_real(x_imag1) + self.dense2_imag(x_real1)
        
        # Final dense layer to N*N
        z_real = self.dense3_real(x_real2) - self.dense3_imag(x_imag2)
        z_imag = self.dense3_real(x_imag2) + self.dense3_imag(x_real2)
        
        # Combine to complex output
        z = tf.complex(z_real, z_imag)
        
        # Reshape last dimension to [N,N]
        new_shape = tf.concat([tf.shape(z)[:-1], [self.N, self.N]], axis=0)
        z = tf.reshape(z, new_shape)
        
        if self.return_IFFT:
            # tile the IDFT matrix for batch
            return tf.tile(self.IDFT_matrix[None,:,:], [new_shape[0], 1, 1])
        else:
            # Compute QQ*
            QQH = tf.matmul(z, tf.linalg.adjoint(z))  # [BATCH, N, N]

            # Sum of diagonal elements
            diag_sum = tf.linalg.trace(QQH)
            diag_sum = tf.sqrt(diag_sum)
            diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
            
            # normalize
            z_normalized = z / diag_sum
            
            # Scale z
            z_scaled = z_normalized * np.sqrt(self.N)

            return z_scaled

if __name__ == "__main__":
    N = 4
    F = idft_matrix(N)
    print(F)