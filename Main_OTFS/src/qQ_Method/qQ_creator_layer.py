import tensorflow as tf
import numpy as np
from keras.layers import Dense, Layer, Conv1D, Conv2D, BatchNormalization, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, GRU
from keras import regularizers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.General_helpers import idft_matrix

class qQ_creator_layer(Layer):
    def __init__(self, N, return_IFFT, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(self.N)  # complex64 matrix
        regularizer = regularizers.l2(1e-4)
        
        # --- New convolutional layers ---
        self.conv1_real = Conv1D(512, kernel_size=128, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.conv1_imag = Conv1D(512, kernel_size=128, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.bn1_real = BatchNormalization()
        self.bn1_imag = BatchNormalization()        
        self.conv2_real = Conv1D(512, kernel_size=64, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.conv2_imag = Conv1D(512, kernel_size=64, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.bn2_real = BatchNormalization()
        self.bn2_imag = BatchNormalization()

        # --- Dense layers (same as before) ---
        self.dense1_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense1_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn3_real = BatchNormalization()
        self.bn3_imag = BatchNormalization()

        self.dense2_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense2_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn4_real = BatchNormalization()
        self.bn4_imag = BatchNormalization()

        self.dense3_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense3_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn5_real = BatchNormalization()
        self.bn5_imag = BatchNormalization()

        self.dense4_real = Dense(N*N + N, use_bias=True, kernel_regularizer=regularizer)
        self.dense4_imag = Dense(N*N + N, use_bias=True, kernel_regularizer=regularizer)
        
        # --- Dropout layers ---
        self.dropout = Dropout(0.05)

    def call(self, h, training=False):
        # Split input into real and imaginary parts
        h_real = tf.math.real(h)
        h_imag = tf.math.imag(h)

        # --- Apply complex Conv1D (2 conv layers) ---
        # Conv1D expects [batch, length, channels]
        # So h_real, h_imag must have 3D shape. If not, we expand dims.
        if len(h_real.shape) == 2:
            h_real = tf.expand_dims(h_real, -1)
            h_imag = tf.expand_dims(h_imag, -1)

        # First complex convolution
        x_real_1D_1 = self.conv1_real(h_real) - self.conv1_imag(h_imag)
        x_imag_1D_1 = self.conv1_real(h_imag) + self.conv1_imag(h_real)
        x_real_1D_1 = self.bn1_real(x_real_1D_1, training=training)
        x_imag_1D_1 = self.bn1_imag(x_imag_1D_1, training=training)
        x_real_1D_1 = tf.nn.relu(x_real_1D_1)
        x_imag_1D_1 = tf.nn.relu(x_imag_1D_1)
        # x_real_1D_1 = self.dropout(x_real_1D_1, training=training)
        # x_imag_1D_1 = self.dropout(x_imag_1D_1, training=training)

        # Second convolution 
        x_real_1D_2  = self.conv2_real(x_real_1D_1) - self.conv2_imag(x_imag_1D_1)
        x_imag_1D_2  = self.conv2_real(x_imag_1D_1) + self.conv2_imag(x_real_1D_1)
        x_real_1D_2  = self.bn2_real(x_real_1D_2, training=training)
        x_imag_1D_2  = self.bn2_imag(x_imag_1D_2, training=training)
        x_real_1D_2  = tf.nn.relu(x_real_1D_2)
        x_imag_1D_2  = tf.nn.relu(x_imag_1D_2)
        # x_real_1D_2 = self.dropout(x_real_1D_2, training=training)
        # x_imag_1D_2 = self.dropout(x_imag_1D_2, training=training)

        # Flatten before Dense layers
        # x_real_dense_0 = tf.reshape(x_real_1D_2, [tf.shape(x_real_1D_2)[0], -1])
        x_real_dense_0 = GlobalAveragePooling1D()(x_real_1D_2)
        # x_imag_dense_0  = tf.reshape(x_imag_1D_2, [tf.shape(x_imag_1D_2)[0], -1])
        x_imag_dense_0 = GlobalAveragePooling1D()(x_imag_1D_2)

        # --- Dense blocks ---
        x_real_dense_1 = self.dense1_real(x_real_dense_0) - self.dense1_imag(x_imag_dense_0)
        x_imag_dense_1 = self.dense1_real(x_imag_dense_0) + self.dense1_imag(x_real_dense_0)
        x_real_dense_1 = self.bn3_real(x_real_dense_1, training=training)
        x_imag_dense_1 = self.bn3_imag(x_imag_dense_1, training=training)
        x_real_dense_1 = tf.nn.relu(x_real_dense_1)
        x_imag_dense_1 = tf.nn.relu(x_imag_dense_1)
        x_real_dense_1 = self.dropout(x_real_dense_1, training=training)
        x_imag_dense_1 = self.dropout(x_imag_dense_1, training=training)

        x_real_dense_2 = self.dense2_real(x_real_dense_1) - self.dense2_imag(x_imag_dense_1) + x_real_dense_1
        x_imag_dense_2 = self.dense2_real(x_imag_dense_1) + self.dense2_imag(x_real_dense_1) + x_imag_dense_1
        x_real_dense_2 = self.bn4_real(x_real_dense_2, training=training)
        x_imag_dense_2 = self.bn4_imag(x_imag_dense_2, training=training)
        x_real_dense_2 = tf.nn.relu(x_real_dense_2)
        x_imag_dense_2 = tf.nn.relu(x_imag_dense_2)
        # x_real_dense_2 = self.dropout(x_real_dense_2, training=training)
        # x_imag_dense_2 = self.dropout(x_imag_dense_2, training=training)

        x_real_dense_3 = self.dense3_real(x_real_dense_2) - self.dense3_imag(x_imag_dense_2) + x_real_dense_2
        x_imag_dense_3 = self.dense3_real(x_imag_dense_2) + self.dense3_imag(x_real_dense_2) + x_imag_dense_2
        x_real_dense_3 = self.bn5_real(x_real_dense_3, training=training)
        x_imag_dense_3 = self.bn5_imag(x_imag_dense_3, training=training)
        x_real_dense_3 = tf.nn.relu(x_real_dense_3)
        x_imag_dense_3 = tf.nn.relu(x_imag_dense_3)

        z_real = self.dense4_real(x_real_dense_3) - self.dense4_imag(x_imag_dense_3)
        z_imag = self.dense4_real(x_imag_dense_3) + self.dense4_imag(x_real_dense_3)

        # Combine to complex output
        z = tf.complex(z_real, z_imag)

        # Reshape to [batch, N, N, 2]
        q = z[..., self.N * self.N:]
        Q = z[..., :self.N * self.N]
        new_shape = tf.concat([tf.shape(z)[:-1], [self.N, self.N]], axis=0)
        Q = tf.reshape(Q, new_shape)

        if self.return_IFFT:
            return tf.tile(self.IDFT_matrix[None, :, :], [new_shape[0], 1, 1]), q
        else:
            QQH = tf.matmul(Q, tf.linalg.adjoint(Q))
            diag_sum = tf.linalg.trace(QQH)
            diag_sum = tf.sqrt(diag_sum)
            diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
            
            Q_normalized = Q / diag_sum
            Q_normalized = Q_normalized * np.sqrt(self.N)

            return Q_normalized, q

class ThetaQ_creator_layer(Layer):
    def __init__(self, N, return_IFFT, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(self.N)  # complex64 matrix
        regularizer = regularizers.l2(1e-4)
        
        # --- New convolutional layers ---
        self.conv1_real = Conv1D(64, kernel_size=128, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.conv1_imag = Conv1D(64, kernel_size=128, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.bn1_real = BatchNormalization()
        self.bn1_imag = BatchNormalization()        
        self.conv2_real = Conv1D(64, kernel_size=64, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.conv2_imag = Conv1D(64, kernel_size=64, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.bn2_real = BatchNormalization()
        self.bn2_imag = BatchNormalization()

        # --- Dense layers (same as before) ---
        self.dense1 = Dense(512, use_bias=False, kernel_regularizer=regularizer)
        self.bn3 = BatchNormalization()

        self.dense2 = Dense(256, use_bias=False, kernel_regularizer=regularizer)
        self.bn4 = BatchNormalization()

        self.dense3 = Dense(128, use_bias=False, kernel_regularizer=regularizer)
        self.bn5 = BatchNormalization()

        self.dense4 = Dense(N, use_bias=True, kernel_regularizer=regularizer)
        
        # --- Dropout layers ---
        self.dropout = Dropout(0.05)

    def call(self, h, training=False):
        # Split input into real and imaginary parts
        h_real = tf.math.real(h)
        h_imag = tf.math.imag(h)

        # --- Apply complex Conv1D (2 conv layers) ---
        # Conv1D expects [batch, length, channels]
        # So h_real, h_imag must have 3D shape. If not, we expand dims.
        if len(h_real.shape) == 2:
            h_real = tf.expand_dims(h_real, -1)
            h_imag = tf.expand_dims(h_imag, -1)

        # First complex convolution
        x_real_1D_1 = self.conv1_real(h_real) - self.conv1_imag(h_imag)
        x_imag_1D_1 = self.conv1_real(h_imag) + self.conv1_imag(h_real)
        x_real_1D_1 = self.bn1_real(x_real_1D_1, training=training)
        x_imag_1D_1 = self.bn1_imag(x_imag_1D_1, training=training)
        x_real_1D_1 = tf.nn.relu(x_real_1D_1)
        x_imag_1D_1 = tf.nn.relu(x_imag_1D_1)

        # Second convolution 
        x_real_1D_2  = self.conv2_real(x_real_1D_1) - self.conv2_imag(x_imag_1D_1)
        x_imag_1D_2  = self.conv2_real(x_imag_1D_1) + self.conv2_imag(x_real_1D_1)
        x_real_1D_2  = self.bn2_real(x_real_1D_2, training=training)
        x_imag_1D_2  = self.bn2_imag(x_imag_1D_2, training=training)
        x_real_1D_2  = tf.nn.relu(x_real_1D_2)
        x_imag_1D_2  = tf.nn.relu(x_imag_1D_2)

        # Flatten before Dense layers
        x = tf.concat([x_real_1D_2,x_imag_1D_2],axis=-1)
        x_dense_0 = tf.reshape(x, [tf.shape(x)[0], -1])

        # --- Dense blocks ---
        x_dense_1 = self.dense1(x_dense_0)
        x_dense_1 = self.bn3(x_dense_1, training=training)
        x_dense_1 = tf.nn.relu(x_dense_1)
        x_dense_1 = self.dropout(x_dense_1, training=training)

        x_dense_2 = self.dense2(x_dense_1)
        x_dense_2 = self.bn4(x_dense_2, training=training)
        x_dense_2 = tf.nn.relu(x_dense_2)

        x_dense_3 = self.dense3(x_dense_2)
        x_dense_3 = self.bn5(x_dense_3, training=training)
        x_dense_3 = tf.nn.relu(x_dense_3)

        z = self.dense4(x_dense_3)

        # Reshape to [batch, N, N, 2]
        thetas = z
        new_shape = tf.concat([tf.shape(z)[:-1], [self.N, self.N]], axis=0)
        
        Q = self.build_F_from_thetas(thetas)
        Q = tf.reshape(Q, new_shape)

        if self.return_IFFT:
            return tf.tile(self.IDFT_matrix[None, :, :], [new_shape[0], 1, 1])
        else:
            QQH = tf.matmul(Q, tf.linalg.adjoint(Q))
            diag_sum = tf.linalg.trace(QQH)
            diag_sum = tf.sqrt(diag_sum)
            diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
            
            Q_normalized = Q / diag_sum
            Q_normalized = Q_normalized * np.sqrt(self.N)

            return Q_normalized

    def build_F_from_thetas(self, thetas):
        """
        thetas: shape [batch_size, n], dtype float32 (radians)
        returns: F of shape [batch_size, n, n], dtype complex64
                where F[b, k, m] = exp(-j * theta[b, k] * m)
        """
        batch_size = tf.shape(thetas)[0]
        n = tf.shape(thetas)[1]

        # m = [0, 1, ..., n-1]
        m = tf.cast(tf.range(n), tf.complex64)  # [n]
        m = tf.reshape(m, [1, 1, n])            # shape [1,1,n] for broadcasting

        # convert to complex
        thetas_c = tf.cast(thetas, tf.complex64)   # [B, n]
        # roots r_k = exp(-j * theta_k)
        roots = tf.exp(-1j * thetas_c)             # [B, n]
        roots = tf.expand_dims(roots, axis=-1)     # [B, n, 1]

        # F[b,k,m] = (roots[b,k]) ** m
        F = tf.pow(roots, m)                       # broadcasting → [B, n, n]
        return F

class OrtQ_creator_layer(Layer):
    def __init__(self, N, return_IFFT, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(self.N)  # complex64 matrix
        regularizer = regularizers.l2(1e-4)
        
        # --- New convolutional layers ---
        self.conv1_real = Conv1D(64, kernel_size=128, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.conv1_imag = Conv1D(64, kernel_size=128, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.bn1_real = BatchNormalization()
        self.bn1_imag = BatchNormalization()        
        self.conv2_real = Conv1D(64, kernel_size=64, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.conv2_imag = Conv1D(64, kernel_size=64, padding='same', use_bias=False, kernel_regularizer=regularizer)
        self.bn2_real = BatchNormalization()
        self.bn2_imag = BatchNormalization()

        # --- Dense layers (same as before) ---
        self.dense1_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense1_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn3_real = BatchNormalization()
        self.bn3_imag = BatchNormalization()

        self.dense2_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense2_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn4_real = BatchNormalization()
        self.bn4_imag = BatchNormalization()

        self.dense3_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense3_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn5_real = BatchNormalization()
        self.bn5_imag = BatchNormalization()

        self.dense4_real = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.dense4_imag = Dense(4048, use_bias=False, kernel_regularizer=regularizer)
        self.bn6_real = BatchNormalization()
        self.bn6_imag = BatchNormalization()

        self.dense5_real = Dense(N * N , use_bias=True, kernel_regularizer=regularizer)
        self.dense5_imag = Dense(N * N , use_bias=True, kernel_regularizer=regularizer)

        # self.dense5_real = Dense(N , use_bias=True, kernel_regularizer=regularizer)
        # self.dense5_imag = Dense(N , use_bias=True, kernel_regularizer=regularizer)

        # --- Dropout layers ---
        self.dropout = Dropout(0.05)

    def call(self, h, training=False):
        # Split input into real and imaginary parts
        h_real = tf.math.real(h)
        h_imag = tf.math.imag(h)

        # --- Apply complex Conv1D (2 conv layers) ---
        # Conv1D expects [batch, length, channels]
        # So h_real, h_imag must have 3D shape. If not, we expand dims.
        if len(h_real.shape) == 2:
            h_real = tf.expand_dims(h_real, -1)
            h_imag = tf.expand_dims(h_imag, -1)

        # First complex convolution
        x_real_1D_1 = self.conv1_real(h_real) - self.conv1_imag(h_imag)
        x_imag_1D_1 = self.conv1_real(h_imag) + self.conv1_imag(h_real)
        x_real_1D_1 = self.bn1_real(x_real_1D_1, training=training)
        x_imag_1D_1 = self.bn1_imag(x_imag_1D_1, training=training)
        x_real_1D_1 = tf.nn.relu(x_real_1D_1)
        x_imag_1D_1 = tf.nn.relu(x_imag_1D_1)

        # Second convolution 
        x_real_1D_2  = self.conv2_real(x_real_1D_1) - self.conv2_imag(x_imag_1D_1)
        x_imag_1D_2  = self.conv2_real(x_imag_1D_1) + self.conv2_imag(x_real_1D_1)
        x_real_1D_2  = self.bn2_real(x_real_1D_2, training=training)
        x_imag_1D_2  = self.bn2_imag(x_imag_1D_2, training=training)
        x_real_1D_2  = tf.nn.relu(x_real_1D_2)
        x_imag_1D_2  = tf.nn.relu(x_imag_1D_2)

        # Flatten before Dense layers
        x_real_dense_0 = tf.reshape(x_real_1D_2, [tf.shape(x_real_1D_2)[0], -1])
        x_imag_dense_0  = tf.reshape(x_imag_1D_2, [tf.shape(x_imag_1D_2)[0], -1])

        # --- Dense blocks ---
        x_real_dense_1 = self.dense1_real(x_real_dense_0) - self.dense1_imag(x_imag_dense_0)
        x_imag_dense_1 = self.dense1_real(x_imag_dense_0) + self.dense1_imag(x_real_dense_0)
        x_real_dense_1 = self.bn3_real(x_real_dense_1, training=training)
        x_imag_dense_1 = self.bn3_imag(x_imag_dense_1, training=training)
        x_real_dense_1 = tf.nn.relu(x_real_dense_1)
        x_imag_dense_1 = tf.nn.relu(x_imag_dense_1)
        x_real_dense_1 = self.dropout(x_real_dense_1, training=training)
        x_imag_dense_1 = self.dropout(x_imag_dense_1, training=training)

        x_real_dense_2 = self.dense2_real(x_real_dense_1) - self.dense2_imag(x_imag_dense_1)
        x_imag_dense_2 = self.dense2_real(x_imag_dense_1) + self.dense2_imag(x_real_dense_1)
        x_real_dense_2 = self.bn4_real(x_real_dense_2, training=training)
        x_imag_dense_2 = self.bn4_imag(x_imag_dense_2, training=training)
        x_real_dense_2 = tf.nn.relu(x_real_dense_2)
        x_imag_dense_2 = tf.nn.relu(x_imag_dense_2)

        x_real_dense_3 = self.dense3_real(x_real_dense_2) - self.dense3_imag(x_imag_dense_2)# + x_real_dense_2
        x_imag_dense_3 = self.dense3_real(x_imag_dense_2) + self.dense3_imag(x_real_dense_2)# + x_imag_dense_2
        x_real_dense_3 = self.bn5_real(x_real_dense_3, training=training)
        x_imag_dense_3 = self.bn5_imag(x_imag_dense_3, training=training)
        x_real_dense_3 = tf.nn.relu(x_real_dense_3)
        x_imag_dense_3 = tf.nn.relu(x_imag_dense_3)

        x_real_dense_4 = self.dense4_real(x_real_dense_3) - self.dense4_imag(x_imag_dense_3)# + x_real_dense_2
        x_imag_dense_4 = self.dense4_real(x_imag_dense_3) + self.dense4_imag(x_real_dense_3)# + x_imag_dense_2
        x_real_dense_4 = self.bn6_real(x_real_dense_4, training=training)
        x_imag_dense_4 = self.bn6_imag(x_imag_dense_4, training=training)
        x_real_dense_4 = tf.nn.relu(x_real_dense_4)
        x_imag_dense_4 = tf.nn.relu(x_imag_dense_4)

        z_real = self.dense5_real(x_real_dense_4) - self.dense5_imag(x_imag_dense_4)
        z_imag = self.dense5_real(x_imag_dense_4) + self.dense5_imag(x_real_dense_4)

        # Combine to complex output
        z = tf.complex(z_real, z_imag)

        # matrix generator
        # Reshape to [batch, N, N, 2]
        new_shape = tf.concat([tf.shape(z)[:-1], [self.N, self.N]], axis=0)
        A = tf.reshape(z, new_shape)
        # create S skew-hermit
        # S = A - tf.linalg.adjoint(A)
        # Exp rais
        # Q = tf.linalg.expm(S)
        # Q=S
        Q=A

        # powers = tf.cast(tf.range(1, self.N + 1), z.dtype)      # (N,)
        # Q = tf.pow(tf.expand_dims(z, 1), tf.reshape(powers, (1, self.N, 1)))

        if self.return_IFFT:
            return tf.tile(tf.cast(self.IDFT_matrix[None, :, :],dtype=tf.complex128), [new_shape[0], 1, 1])
        else:
            QQH = tf.matmul(Q, tf.linalg.adjoint(Q))
            diag_sum = tf.linalg.trace(QQH)
            diag_sum = tf.sqrt(diag_sum)
            diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
            
            Q_normalized = Q / diag_sum
            Q_normalized = Q_normalized * np.sqrt(self.N)

            return Q_normalized

class TimePositionalEncoding(Layer):
    """
    Sinusoidal positional encoding along the time (snapshot) dimension.

    Injects explicit time-position information into each CIR snapshot so the
    network can distinguish different OFDM symbol positions and learn
    time-varying (Doppler) channel patterns.

    For delay-tap index i and snapshot index j:
        PE[i, 2k]   = sin(j / 10000^(2k / d_model))
        PE[i, 2k+1] = cos(j / 10000^(2k+1 / d_model))
    where d_model = l_tot (number of delay taps).
    """
    def __init__(self, pe_scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.pe_scale = pe_scale

    def call(self, h_real, h_imag):
        """
        Args:
            h_real: (batch, l_tot, num_snapshots) float32
            h_imag: (batch, l_tot, num_snapshots) float32
        Returns:
            h_real, h_imag with positional encoding added (same shape)
        """
        l_tot = tf.shape(h_real)[1]
        num_snapshots = tf.shape(h_real)[2]

        # Position indices: [0, 1, ..., num_snapshots-1]
        pos = tf.range(num_snapshots, dtype=tf.float32)  # (num_snapshots,)
        # Feature indices: [0, 1, ..., l_tot-1]
        feat = tf.range(l_tot, dtype=tf.float32)  # (l_tot,)

        # div_term[i] = 10000^(i / l_tot) → shape (l_tot,)
        div_term = tf.pow(10000.0, feat / tf.cast(l_tot, tf.float32))

        # angle[i, j] = pos[j] / div_term[i] → shape (l_tot, num_snapshots)
        angle = tf.expand_dims(1.0 / div_term, 1) * tf.expand_dims(pos, 0)

        # Interleave sin/cos: even rows = sin, odd rows = cos
        even_mask = tf.cast(tf.range(l_tot) % 2 == 0, tf.float32)  # (l_tot,)
        odd_mask = 1.0 - even_mask
        pe = (tf.sin(angle) * tf.expand_dims(even_mask, 1)
              + tf.cos(angle) * tf.expand_dims(odd_mask, 1))  # (l_tot, num_snapshots)

        pe = tf.expand_dims(pe, 0)  # (1, l_tot, num_snapshots)

        # Add PE to both real and imaginary parts
        return (h_real + pe * self.pe_scale,
                h_imag + pe * self.pe_scale)


class qQ_creator_conv_gru(Layer):
    """
    Dual-path CIR processor with explicit time (Doppler) modeling.

    Path A — Delay-dim: Conv1D + GRU along delay taps (multipath structure).
    Path B — Time-dim:  Transpose → Conv1D along delay-features → Conv1D along
                time (cross-snapshot) → pool (Doppler/time-evolution signature).

    The two paths are fused before the feedforward block so the network can
    jointly reason about multipath structure AND channel time-variation.
    """
    def __init__(self, N, conv_filters=64, conv_kernel=32, gru_units=1024,
                 ff_dim=1024, dropout_rate=0.1, num_gru_layers=1, return_IFFT=False):
        super().__init__()
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(N)
        reg = regularizers.l2(1e-4)
        self.N = N

        # --- Time positional encoding (increased scale for stronger time signal) ---
        self.time_pe = TimePositionalEncoding(pe_scale=0.10)

        # ========================
        # Path A: Delay-dim complex Conv1D + GRU
        # ========================
        self.conv_real = Conv1D(conv_filters, conv_kernel, padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv_imag = Conv1D(conv_filters, conv_kernel, padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_real = BatchNormalization()
        self.bn_imag = BatchNormalization()

        self.gru_layers = []
        for _ in range(num_gru_layers):
            self.gru_layers.append(ComplexGRUCell(gru_units))

        # ========================
        # Path B: Time-dim processing (explicit Doppler modeling)
        # ========================
        time_features = 64
        # Conv1D along delay-features (after transpose, l_tot becomes feature dim)
        self.conv_time1 = Conv1D(time_features, 16, padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_time1 = BatchNormalization()
        # Conv1D along TIME axis (kernel=3 covers all 3 snapshots → cross-snapshot interaction)
        self.conv_time2 = Conv1D(time_features, 3, padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_time2 = BatchNormalization()

        # ========================
        # Fusion: project concatenated paths back to gru_units
        # ========================
        self.fuse_real = Dense(gru_units, use_bias=False, kernel_regularizer=reg)
        self.fuse_imag = Dense(gru_units, use_bias=False, kernel_regularizer=reg)
        self.bn_fuse_real = BatchNormalization()
        self.bn_fuse_imag = BatchNormalization()

        # ========================
        # Feedforward + Output (same as original)
        # ========================
        self.ff1 = Dense(ff_dim, activation='relu', kernel_regularizer=reg)
        self.ff2 = Dense(gru_units, kernel_regularizer=reg)
        self.dropout = Dropout(dropout_rate)

        self.dense_qQ_real = Dense(N*N + N, kernel_regularizer=reg)
        self.dense_qQ_imag = Dense(N*N + N, kernel_regularizer=reg)
        self.dense_q_real = Dense(N, kernel_regularizer=reg)
        self.dense_q_imag = Dense(N, kernel_regularizer=reg)

    def call(self, h, training=False):
        # h shape: (batch, l_tot, num_snapshots) complex

        # ================================================================
        # Step 1: Add Time Positional Encoding
        # ================================================================
        h_real_raw = tf.math.real(h)
        h_imag_raw = tf.math.imag(h)
        h_real, h_imag = self.time_pe(h_real_raw, h_imag_raw)
        h_pe = tf.complex(h_real, h_imag)

        # ================================================================
        # Path A: Delay-dim processing (multipath structure)
        #   (batch, l_tot, 3) → Conv1D → (batch, l_tot, 64) → GRU → (batch, l_tot, gru_units)
        # ================================================================
        x_delay_real = self.bn_real(
            self.conv_real(tf.math.real(h_pe)) - self.conv_imag(tf.math.imag(h_pe)),
            training=training)
        x_delay_imag = self.bn_imag(
            self.conv_real(tf.math.imag(h_pe)) + self.conv_imag(tf.math.real(h_pe)),
            training=training)
        x_delay = tf.complex(x_delay_real, x_delay_imag)

        for gru in self.gru_layers:
            x_delay = gru(x_delay)  # (batch, l_tot, gru_units)

        # ================================================================
        # Path B: Time-dim processing (Doppler / time-evolution signature)
        #   Transpose → (batch, 3, l_tot)
        #   → Conv1D(l_tot features) → Conv1D(cross-time, kernel=3) → pool
        # ================================================================
        h_time = tf.transpose(h_pe, [0, 2, 1])  # (batch, num_snapshots, l_tot)
        # Concat real+imag along feature dim: (batch, 3, 2*l_tot)
        x_time = tf.concat([tf.math.real(h_time), tf.math.imag(h_time)], axis=-1)

        # Extract delay-feature patterns per time step
        x_time = self.bn_time1(self.conv_time1(x_time), training=training)
        x_time = tf.nn.relu(x_time)  # (batch, 3, time_features)

        # Cross-snapshot Conv1D: models interaction between adjacent time snapshots
        x_time = self.bn_time2(self.conv_time2(x_time), training=training)
        x_time = tf.nn.relu(x_time)  # (batch, 3, time_features)

        # Pool over time → global Doppler signature
        x_time_pool = tf.reduce_mean(x_time, axis=1)  # (batch, time_features)

        # Broadcast time-path features to every delay tap
        l_tot = tf.shape(x_delay_real)[1]
        x_time_bc = tf.tile(
            tf.expand_dims(x_time_pool, 1), [1, l_tot, 1])  # (batch, l_tot, time_features)

        # ================================================================
        # Fusion: concat delay + time paths, project to gru_units
        # ================================================================
        x_fuse_real = tf.concat([tf.math.real(x_delay), x_time_bc], axis=-1)
        x_fuse_imag = tf.concat([tf.math.imag(x_delay), tf.zeros_like(x_time_bc)], axis=-1)

        x_real = self.bn_fuse_real(
            self.fuse_real(x_fuse_real) - self.fuse_imag(x_fuse_imag),
            training=training)
        x_imag = self.bn_fuse_imag(
            self.fuse_real(x_fuse_imag) + self.fuse_imag(x_fuse_real),
            training=training)

        # ================================================================
        # Feedforward + Global pooling + Output (same as original)
        # ================================================================
        ff_real = self.ff2(self.ff1(x_real))
        ff_imag = self.ff2(self.ff1(x_imag))
        x_real = self.dropout(x_real + ff_real, training=training)
        x_imag = self.dropout(x_imag + ff_imag, training=training)

        x_real = tf.reduce_mean(x_real, axis=1)
        x_imag = tf.reduce_mean(x_imag, axis=1)

        qQ_real = self.dense_qQ_real(x_real) - self.dense_qQ_imag(x_imag)
        qQ_imag = self.dense_qQ_real(x_imag) + self.dense_qQ_imag(x_real)
        qQ = tf.complex(qQ_real, qQ_imag)
        q = qQ[..., self.N*self.N:]
        Q = qQ[..., :self.N*self.N]
        Q = tf.reshape(Q, (-1, self.N, self.N))

        q_real = self.dense_q_real(tf.math.real(q)) - self.dense_q_imag(tf.math.imag(q))
        q_imag = self.dense_q_real(tf.math.real(q)) + self.dense_q_imag(tf.math.imag(q))
        q = tf.complex(q_real, q_imag)

        # --- Normalize Q ---
        QQH = tf.matmul(Q, tf.linalg.adjoint(Q))
        diag_sum = tf.linalg.trace(QQH)
        diag_sum = tf.sqrt(diag_sum)
        diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
        Q_normalized = Q / diag_sum
        Q_normalized = Q_normalized * np.sqrt(self.N)

        if self.return_IFFT:
            idft_matrix = tf.tile(self.IDFT_matrix[None, :, :], [Q_normalized.shape[0], 1, 1])
            return idft_matrix, q
        else:
            return Q_normalized, q

class qQ_creator_conv2d(Layer):
    """
    2D Conv CIR processor — Plan B: shallow conv + structured pooling.

    Key changes from original:
      - 2 Conv2D blocks instead of 4 → receptive field ~5×5 on ~10×12 input
        → neurons have LOCAL spatial specialization, can distinguish delay positions
      - Structured pooling: separate statistics for delay structure (multipath)
        and time variation (Doppler), instead of global mean
      - Input: (batch, l_tot, num_snapshots) complex + Time PE

    Architecture:
      Input → 2× Complex Conv2D (3×3, 32→64 ch) → Structured pooling → Dense head → Q+q
    """
    def __init__(self, N, return_IFFT=False, dropout_rate=0.1):
        super().__init__()
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(N)
        reg = regularizers.l2(1e-4)
        self.N = N

        # --- Time positional encoding ---
        self.time_pe = TimePositionalEncoding(pe_scale=0.10)

        # ========================
        # Complex Conv2D Block 1:  1 → 32
        # ========================
        self.conv2d_1r = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv2d_1i = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_1r = BatchNormalization()
        self.bn_1i = BatchNormalization()

        # ========================
        # Complex Conv2D Block 2:  32 → 64
        # ========================
        self.conv2d_2r = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv2d_2i = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_2r = BatchNormalization()
        self.bn_2i = BatchNormalization()

        # ========================
        # Structured pooling → concat to 64*4 = 256 dims
        # Project 256 → 512
        # ========================
        self.proj_real = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.proj_imag = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.bn_proj_real = BatchNormalization()
        self.bn_proj_imag = BatchNormalization()

        self.dense1_real = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.dense1_imag = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.bn_d1_real = BatchNormalization()
        self.bn_d1_imag = BatchNormalization()

        self.dense2_real = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.dense2_imag = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.bn_d2_real = BatchNormalization()
        self.bn_d2_imag = BatchNormalization()

        self.dense_out_real = Dense(N * N + N, kernel_regularizer=reg)
        self.dense_out_imag = Dense(N * N + N, kernel_regularizer=reg)

        self.dropout = Dropout(dropout_rate)

    def _complex_conv2d_block(self, x_real, x_imag, conv_r, conv_i, bn_r, bn_i, training):
        out_real = conv_r(x_real) - conv_i(x_imag)
        out_imag = conv_r(x_imag) + conv_i(x_real)
        out_real = bn_r(out_real, training=training)
        out_imag = bn_i(out_imag, training=training)
        return tf.nn.relu(out_real), tf.nn.relu(out_imag)

    def call(self, h, training=False):
        # h shape: (batch, l_tot, num_snapshots) complex

        # ---- Time Positional Encoding ----
        h_real, h_imag = self.time_pe(tf.math.real(h), tf.math.imag(h))

        # ---- Reshape for Conv2D: (batch, l_tot, num_snapshots, 1) ----
        h_real = tf.expand_dims(h_real, -1)
        h_imag = tf.expand_dims(h_imag, -1)

        # ---- Complex Conv2D Blocks (only 2 → receptive field ~5×5) ----
        x_real, x_imag = self._complex_conv2d_block(
            h_real, h_imag, self.conv2d_1r, self.conv2d_1i, self.bn_1r, self.bn_1i, training)
        # (batch, l_tot, num_snapshots, 32)

        x_real, x_imag = self._complex_conv2d_block(
            x_real, x_imag, self.conv2d_2r, self.conv2d_2i, self.bn_2r, self.bn_2i, training)
        # (batch, l_tot, num_snapshots, 64)

        # ================================================================
        # Structured pooling — PRESERVE spatial information
        # ================================================================

        # Stat 1: Mean over TIME per delay tap → multipath structure
        #   "For each delay tap, what's the average activation?"
        delay_mean_r = tf.reduce_mean(x_real, axis=2)  # (batch, l_tot, 64)
        delay_mean_i = tf.reduce_mean(x_imag, axis=2)
        dm_r = tf.reduce_mean(delay_mean_r, axis=1)     # (batch, 64)
        dm_i = tf.reduce_mean(delay_mean_i, axis=1)

        # Stat 2: Variance over TIME per delay tap → Doppler indicator
        #   "How much does each delay tap vary across snapshots?"
        time_var_r = tf.math.reduce_variance(x_real, axis=2)  # (batch, l_tot, 64)
        time_var_i = tf.math.reduce_variance(x_imag, axis=2)
        tv_r = tf.reduce_mean(time_var_r, axis=1)              # (batch, 64)
        tv_i = tf.reduce_mean(time_var_i, axis=1)

        # Stat 3: Std over DELAY per snapshot → multipath richness
        #   "How spread out is energy across delay taps?"
        #   Use sqrt(var + eps) to avoid NaN gradients when var≈0
        time_mean_r = tf.reduce_mean(x_real, axis=1)  # (batch, num_snapshots, 64)
        time_mean_i = tf.reduce_mean(x_imag, axis=1)
        tm_var_r = tf.math.reduce_variance(time_mean_r, axis=1)  # (batch, 64)
        tm_var_i = tf.math.reduce_variance(time_mean_i, axis=1)
        tm_std_r = tf.sqrt(tm_var_r + 1e-8)  # numerically stable std
        tm_std_i = tf.sqrt(tm_var_i + 1e-8)

        # Stat 4: Global mean (baseline)
        global_r = tf.reduce_mean(x_real, axis=[1, 2])  # (batch, 64)
        global_i = tf.reduce_mean(x_imag, axis=[1, 2])

        # ---- Concatenate: 64×4 = 256 dims ----
        x_real = tf.concat([dm_r, tv_r, tm_std_r, global_r], axis=-1)  # (batch, 256)
        x_imag = tf.concat([dm_i, tv_i, tm_std_i, global_i], axis=-1)

        # ---- Projection: 256 → 512 ----
        p_real = self.proj_real(x_real) - self.proj_imag(x_imag)
        p_imag = self.proj_real(x_imag) + self.proj_imag(x_real)
        p_real = self.bn_proj_real(p_real, training=training)
        p_imag = self.bn_proj_imag(p_imag, training=training)
        p_real = tf.nn.relu(p_real)
        p_imag = tf.nn.relu(p_imag)

        # ---- Complex Dense Block 1 ----
        d1_real = self.dense1_real(p_real) - self.dense1_imag(p_imag)
        d1_imag = self.dense1_real(p_imag) + self.dense1_imag(p_real)
        d1_real = self.bn_d1_real(d1_real, training=training)
        d1_imag = self.bn_d1_imag(d1_imag, training=training)
        d1_real = tf.nn.relu(d1_real)
        d1_imag = tf.nn.relu(d1_imag)
        d1_real = self.dropout(d1_real, training=training)
        d1_imag = self.dropout(d1_imag, training=training)

        # ---- Complex Dense Block 2 (with residual) ----
        d2_real = self.dense2_real(d1_real) - self.dense2_imag(d1_imag)
        d2_imag = self.dense2_real(d1_imag) + self.dense2_imag(d1_real)
        d2_real = self.bn_d2_real(d2_real, training=training)
        d2_imag = self.bn_d2_imag(d2_imag, training=training)
        d2_real = tf.nn.relu(d2_real + d1_real)
        d2_imag = tf.nn.relu(d2_imag + d1_imag)

        # ---- Output head ----
        out_real = self.dense_out_real(d2_real) - self.dense_out_imag(d2_imag)
        out_imag = self.dense_out_real(d2_imag) + self.dense_out_imag(d2_real)
        z = tf.complex(out_real, out_imag)

        q = z[..., self.N * self.N:]
        Q = z[..., :self.N * self.N]
        Q = tf.reshape(Q, (-1, self.N, self.N))

        # ---- Normalize Q ----
        QQH = tf.matmul(Q, tf.linalg.adjoint(Q))
        diag_sum = tf.linalg.trace(QQH)
        diag_sum = tf.sqrt(diag_sum)
        diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
        Q_normalized = Q / diag_sum
        Q_normalized = Q_normalized * tf.sqrt(tf.cast(self.N, Q.dtype))

        if self.return_IFFT:
            idft_tiled = tf.tile(self.IDFT_matrix[None, :, :], [Q_normalized.shape[0], 1, 1])
            return tf.cast(idft_tiled, Q_normalized.dtype), q
        else:
            return Q_normalized, q


class ComplexGRUCell(Layer):
    """A simple complex GRU cell operating on tf.complex64 inputs."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.real_gru = GRU(units, return_sequences=True)
        self.imag_gru = GRU(units, return_sequences=True)

    def call(self, inputs):
        x_real = tf.math.real(inputs)
        x_imag = tf.math.imag(inputs)
        out_real = self.real_gru(x_real) - self.imag_gru(x_imag)
        out_imag = self.real_gru(x_imag) + self.imag_gru(x_real)
        return tf.complex(out_real, out_imag)


class qQ_creator_conv2d_v2(Layer):
    """
    2D Conv CIR processor — 结构化池化保留时延+多普勒信息。

    关键改进 vs qQ_creator_conv2d:
      1. 分离池化: delay维池化 → 多普勒特征, time维池化 → 多径结构, 全局池化 → 整体
      2. 更强的 Time PE (scale=0.5 → 1.0)
      3. 三路特征融合 → 密集层 → Q矩阵

    这样网络可以根据 CIR 区分:
      - 平坦信道 (delay维无结构) → Q ≈ I (TDM)
      - 多径+低速 (delay有结构, time平稳) → Q ≈ IDFT (OFDM)
      - 多径+高速 (delay有结构, time变化) → Q 是密集扩展矩阵 (OTFS)
    """
    def __init__(self, N, return_IFFT=False, dropout_rate=0.1):
        super().__init__()
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(N)
        reg = regularizers.l2(1e-4)
        self.N = N

        # --- Stronger time positional encoding ---
        self.time_pe = TimePositionalEncoding(pe_scale=0.5)

        # ========================
        # Complex Conv2D blocks (same as v1 for feature extraction)
        # ========================
        self.conv2d_1r = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv2d_1i = Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_1r = BatchNormalization()
        self.bn_1i = BatchNormalization()

        self.conv2d_2r = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv2d_2i = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_2r = BatchNormalization()
        self.bn_2i = BatchNormalization()

        self.conv2d_3r = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv2d_3i = Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_3r = BatchNormalization()
        self.bn_3i = BatchNormalization()

        self.conv2d_4r = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv2d_4i = Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_4r = BatchNormalization()
        self.bn_4i = BatchNormalization()

        # ========================
        # 三路池化后的投影层
        # ========================
        # Path A: delay池化 → 多普勒特征 (time维保留) → 256 → 512
        self.pathA_proj_r = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.pathA_proj_i = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.bnA_r = BatchNormalization()
        self.bnA_i = BatchNormalization()

        # Path B: time池化 → 多径结构 (delay维保留) → 256 → 512
        self.pathB_proj_r = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.pathB_proj_i = Dense(512, use_bias=False, kernel_regularizer=reg)
        self.bnB_r = BatchNormalization()
        self.bnB_i = BatchNormalization()

        # Path C: 全局池化 → 整体特征 → 256 → 256
        self.pathC_proj_r = Dense(256, use_bias=False, kernel_regularizer=reg)
        self.pathC_proj_i = Dense(256, use_bias=False, kernel_regularizer=reg)
        self.bnC_r = BatchNormalization()
        self.bnC_i = BatchNormalization()

        # ========================
        # 融合层: 512+512+256 = 1280 → 1024
        # ========================
        self.fuse_r = Dense(1024, use_bias=False, kernel_regularizer=reg)
        self.fuse_i = Dense(1024, use_bias=False, kernel_regularizer=reg)
        self.bn_fuse_r = BatchNormalization()
        self.bn_fuse_i = BatchNormalization()

        # ========================
        # Dense head
        # ========================
        self.dense1_real = Dense(1024, use_bias=False, kernel_regularizer=reg)
        self.dense1_imag = Dense(1024, use_bias=False, kernel_regularizer=reg)
        self.bn_d1_real = BatchNormalization()
        self.bn_d1_imag = BatchNormalization()

        self.dense2_real = Dense(1024, use_bias=False, kernel_regularizer=reg)
        self.dense2_imag = Dense(1024, use_bias=False, kernel_regularizer=reg)
        self.bn_d2_real = BatchNormalization()
        self.bn_d2_imag = BatchNormalization()

        self.dense_out_real = Dense(N * N + N, kernel_regularizer=reg)
        self.dense_out_imag = Dense(N * N + N, kernel_regularizer=reg)

        self.dropout = Dropout(dropout_rate)

    def _complex_conv2d_block(self, x_real, x_imag, conv_r, conv_i, bn_r, bn_i, training):
        out_real = conv_r(x_real) - conv_i(x_imag)
        out_imag = conv_r(x_imag) + conv_i(x_real)
        out_real = bn_r(out_real, training=training)
        out_imag = bn_i(out_imag, training=training)
        return tf.nn.relu(out_real), tf.nn.relu(out_imag)

    def _complex_dense(self, x_real, x_imag, dense_r, dense_i, bn_r, bn_i, training, activation=True):
        out_real = dense_r(x_real) - dense_i(x_imag)
        out_imag = dense_r(x_imag) + dense_i(x_real)
        out_real = bn_r(out_real, training=training)
        out_imag = bn_i(out_imag, training=training)
        if activation:
            out_real = tf.nn.relu(out_real)
            out_imag = tf.nn.relu(out_imag)
        return out_real, out_imag

    def call(self, h, training=False):
        # h shape: (batch, l_tot, num_snapshots) complex

        # ---- Time Positional Encoding (stronger scale) ----
        h_real, h_imag = self.time_pe(tf.math.real(h), tf.math.imag(h))

        # ---- Reshape for Conv2D: (batch, l_tot, num_snapshots, 1) ----
        h_real = tf.expand_dims(h_real, -1)
        h_imag = tf.expand_dims(h_imag, -1)

        # ---- Complex Conv2D Blocks ----
        x_real, x_imag = self._complex_conv2d_block(
            h_real, h_imag, self.conv2d_1r, self.conv2d_1i, self.bn_1r, self.bn_1i, training)
        x_real, x_imag = self._complex_conv2d_block(
            x_real, x_imag, self.conv2d_2r, self.conv2d_2i, self.bn_2r, self.bn_2i, training)
        x_real, x_imag = self._complex_conv2d_block(
            x_real, x_imag, self.conv2d_3r, self.conv2d_3i, self.bn_3r, self.bn_3i, training)
        x_real, x_imag = self._complex_conv2d_block(
            x_real, x_imag, self.conv2d_4r, self.conv2d_4i, self.bn_4r, self.bn_4i, training)
        # (batch, l_tot, num_snapshots, 256)

        # ================================================================
        # 三路结构化池化（替代 GlobalAveragePooling2D）
        # ================================================================

        # Path A: 对 delay 维池化 → 保留 time 维 → 多普勒特征
        #   (batch, l_tot, num_snapshots, 256) → mean over delay → (batch, num_snapshots, 256)
        pathA_real = tf.reduce_mean(x_real, axis=1)
        pathA_imag = tf.reduce_mean(x_imag, axis=1)

        # Path B: 对 time 维池化 → 保留 delay 维 → 多径结构
        #   (batch, l_tot, num_snapshots, 256) → mean over time → (batch, l_tot, 256)
        pathB_real = tf.reduce_mean(x_real, axis=2)
        pathB_imag = tf.reduce_mean(x_imag, axis=2)

        # Path C: 全局池化 → 整体特征 (作为 baseline)
        #   (batch, l_tot, num_snapshots, 256) → mean over both → (batch, 256)
        pathC_real = tf.reduce_mean(x_real, axis=[1, 2])
        pathC_imag = tf.reduce_mean(x_imag, axis=[1, 2])

        # ---- 各路独立投影 ----
        pathA_real, pathA_imag = self._complex_dense(
            pathA_real, pathA_imag, self.pathA_proj_r, self.pathA_proj_i,
            self.bnA_r, self.bnA_i, training)
        # (batch, num_snapshots, 512)

        pathB_real, pathB_imag = self._complex_dense(
            pathB_real, pathB_imag, self.pathB_proj_r, self.pathB_proj_i,
            self.bnB_r, self.bnB_i, training)
        # (batch, l_tot, 512)

        pathC_real, pathC_imag = self._complex_dense(
            pathC_real, pathC_imag, self.pathC_proj_r, self.pathC_proj_i,
            self.bnC_r, self.bnC_i, training)
        # (batch, 256)

        # ---- 各路再池化到 1D 向量 ----
        pathA_real = tf.reduce_mean(pathA_real, axis=1)  # (batch, 512)
        pathA_imag = tf.reduce_mean(pathA_imag, axis=1)  # (batch, 512)
        pathB_real = tf.reduce_mean(pathB_real, axis=1)  # (batch, 512)
        pathB_imag = tf.reduce_mean(pathB_imag, axis=1)  # (batch, 512)

        # ---- 三路拼接 ----
        fuse_real = tf.concat([pathA_real, pathB_real, pathC_real], axis=-1)  # (batch, 1280)
        fuse_imag = tf.concat([pathA_imag, pathB_imag, pathC_imag], axis=-1)  # (batch, 1280)

        # ---- 融合投影: 1280 → 1024 ----
        fuse_real, fuse_imag = self._complex_dense(
            fuse_real, fuse_imag, self.fuse_r, self.fuse_i,
            self.bn_fuse_r, self.bn_fuse_i, training)

        # ---- Dense Head (same as v1) ----
        d1_real, d1_imag = self._complex_dense(
            fuse_real, fuse_imag, self.dense1_real, self.dense1_imag,
            self.bn_d1_real, self.bn_d1_imag, training)
        d1_real = self.dropout(d1_real, training=training)
        d1_imag = self.dropout(d1_imag, training=training)

        d2_real, d2_imag = self._complex_dense(
            d1_real, d1_imag, self.dense2_real, self.dense2_imag,
            self.bn_d2_real, self.bn_d2_imag, training, activation=False)
        d2_real = tf.nn.relu(d2_real + d1_real)
        d2_imag = tf.nn.relu(d2_imag + d1_imag)

        # ---- Output head ----
        out_real = self.dense_out_real(d2_real) - self.dense_out_imag(d2_imag)
        out_imag = self.dense_out_real(d2_imag) + self.dense_out_imag(d2_real)
        z = tf.complex(out_real, out_imag)

        q = z[..., self.N * self.N:]
        Q = z[..., :self.N * self.N]
        Q = tf.reshape(Q, (-1, self.N, self.N))

        # ---- Normalize Q ----
        QQH = tf.matmul(Q, tf.linalg.adjoint(Q))
        diag_sum = tf.linalg.trace(QQH)
        diag_sum = tf.sqrt(diag_sum)
        diag_sum = tf.reshape(diag_sum, (-1, 1, 1))
        Q_normalized = Q / diag_sum
        Q_normalized = Q_normalized * tf.sqrt(tf.cast(self.N, Q.dtype))

        if self.return_IFFT:
            idft_tiled = tf.tile(self.IDFT_matrix[None, :, :], [Q_normalized.shape[0], 1, 1])
            return tf.cast(idft_tiled, Q_normalized.dtype), q
        else:
            return Q_normalized, q
    
if __name__ == "__main__":
    N = 4
    F = idft_matrix(N)
    print(F)