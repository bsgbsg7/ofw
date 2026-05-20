import tensorflow as tf
import numpy as np
from keras.layers import Dense, Layer, Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, GRU
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

class qQ_creator_conv_gru(Layer):
    def __init__(self, N, conv_filters=64, conv_kernel=32, gru_units=1024,
                 ff_dim=1024, dropout_rate=0.1, num_gru_layers=1, return_IFFT=False):
        super().__init__()
        self.return_IFFT = return_IFFT
        self.IDFT_matrix = idft_matrix(N)
        reg = regularizers.l2(1e-4)
        self.N = N

        # --- Complex Conv1D layers ---
        self.conv_real = Conv1D(conv_filters, conv_kernel, padding='same', use_bias=False, kernel_regularizer=reg)
        self.conv_imag = Conv1D(conv_filters, conv_kernel, padding='same', use_bias=False, kernel_regularizer=reg)
        self.bn_real = BatchNormalization()
        self.bn_imag = BatchNormalization()

        # --- Complex GRU layers ---
        self.gru_layers = []
        for _ in range(num_gru_layers):
            self.gru_layers.append(ComplexGRUCell(gru_units))

        # --- Feedforward ---
        self.ff1 = Dense(ff_dim, activation='relu', kernel_regularizer=reg)
        self.ff2 = Dense(gru_units, kernel_regularizer=reg)
        self.dropout = Dropout(dropout_rate)

        # --- Output heads ---
        self.dense_qQ_real = Dense(N*N + N, kernel_regularizer=reg)
        self.dense_qQ_imag = Dense(N*N + N, kernel_regularizer=reg)
        self.dense_q_real = Dense(N, kernel_regularizer=reg)
        self.dense_q_imag = Dense(N, kernel_regularizer=reg)

    def call(self, h, training=False):
        # --- Complex Conv1D ---
        x_real = self.bn_real(self.conv_real(tf.math.real(h)) - self.conv_imag(tf.math.imag(h)), training=training)
        x_imag = self.bn_imag(self.conv_real(tf.math.imag(h)) + self.conv_imag(tf.math.real(h)), training=training)
        x = tf.complex(x_real, x_imag)

        # --- Complex GRU ---
        for gru in self.gru_layers:
            x = gru(x)

        # --- Feedforward ---
        ff_real = self.ff2(self.ff1(tf.math.real(x)))
        ff_imag = self.ff2(self.ff1(tf.math.imag(x)))
        x_real = self.dropout(tf.math.real(x) + ff_real, training=training)
        x_imag = self.dropout(tf.math.imag(x) + ff_imag, training=training)

        # --- Global pooling ---
        x_real = tf.reduce_mean(x_real, axis=1)
        x_imag = tf.reduce_mean(x_imag, axis=1)

        # --- Output heads ---
        qQ_real = self.dense_qQ_real(x_real) - self.dense_qQ_imag(x_imag)
        qQ_imag = self.dense_qQ_real(x_imag) + self.dense_qQ_imag(x_real)
        qQ = tf.complex(qQ_real, qQ_imag)
        q = qQ[...,self.N*self.N:]
        Q = qQ[...,:self.N*self.N]
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
    
if __name__ == "__main__":
    N = 4
    F = idft_matrix(N)
    print(F)