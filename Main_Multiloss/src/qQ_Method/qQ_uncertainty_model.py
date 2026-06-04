import tensorflow as tf
from keras.layers import Dense, Layer, Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, GRU
from keras import regularizers
from keras.layers import Dense, Layer

class UncertaintyModel_2D(Layer):
    def __init__(
        self,
        hidden_units = 256,
        min_log_sigma = -10.0,
        max_log_sigma = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        reg = regularizers.l2(1e-4)
        self.ff1 = Dense(hidden_units, activation='relu', kernel_regularizer=reg)
        self.bn1 = BatchNormalization()
        self.ff2 = Dense(hidden_units*2, activation='relu', kernel_regularizer=reg)
        self.bn2 = BatchNormalization()
        self.ff3 = Dense(3, kernel_regularizer=reg)
        self.dropout = Dropout(0.2)


    def call(self, channel_features, training=False):

        out_1 = self.ff1(channel_features)
        out_1 = self.bn1(out_1, training=training)
        out_1 = self.dropout(out_1, training=training)
        out_2 = self.ff2(out_1)
        out_2 = self.bn2(out_2, training=training)
        log_sigma = self.ff3(out_2)
        log_sigma = tf.clip_by_value(
                                                log_sigma,
                                                self.min_log_sigma,
                                                self.max_log_sigma
                                            )
        # log_sigma = tf.squeeze(log_sigma)
        log_sigma_par = log_sigma[:,0]
        log_sigma_bce = log_sigma[:,1]
        log_sigma_par_lim = log_sigma[:,2]
        return log_sigma_par, log_sigma_bce, log_sigma_par_lim
    
class UncertaintyModel_1D(Layer):
    def __init__(
        self,
        hidden_units = 256,
        min_log_sigma = 2.0,
        max_log_sigma = 6.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        reg = regularizers.l2(1e-4)
        self.ff1 = Dense(hidden_units, activation='relu', kernel_regularizer=reg)
        self.bn1 = BatchNormalization()
        self.ff2 = Dense(hidden_units*2, activation='relu', kernel_regularizer=reg)
        self.bn2 = BatchNormalization()
        self.ff3 = Dense(1, activation='sigmoid')
        self.dropout = Dropout(0.2)


    def call(self, channel_features, training=False):

        out_1 = self.ff1(channel_features)
        out_1 = self.bn1(out_1, training=training)
        out_1 = self.dropout(out_1, training=training)
        out_2 = self.ff2(out_1)
        out_2 = self.bn2(out_2, training=training)
        log_sigma = self.ff3(out_2) * self.max_log_sigma
        log_sigma = tf.clip_by_value(
                                                log_sigma,
                                                self.min_log_sigma,
                                                self.max_log_sigma
                                            )
        log_sigma = tf.squeeze(log_sigma)
        return log_sigma

class UncertaintyModel_4D(Layer):
    """
    4-output Uncertainty Network for multi-task loss weighting.

    Outputs logσ² for each of the 4 loss terms:
      - logσ²_bce:  uncertainty for BCE (bit error rate)
      - logσ²_papr: uncertainty for PAPR (peak-to-average power ratio)
      - logσ²_oob:  uncertainty for OOB (out-of-band emission)
      - logσ²_af:   uncertainty for AF Shape (ambiguity function)

    Kendall et al. 2018 formulation:
      L_total = Σ_i [ exp(-logσ²_i) · L_i + logσ²_i ]

    The logσ² values are clipped to [-5, 5] for training stability.
    """
    def __init__(
        self,
        hidden_units=256,
        min_log_sigma=-5.0,
        max_log_sigma=5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        reg = regularizers.l2(1e-4)
        self.ff1 = Dense(hidden_units, activation='relu', kernel_regularizer=reg)
        self.bn1 = BatchNormalization()
        self.ff2 = Dense(hidden_units*2, activation='relu', kernel_regularizer=reg)
        self.bn2 = BatchNormalization()
        self.ff3 = Dense(4, kernel_regularizer=reg)  # 4 log-sigma outputs
        self.dropout = Dropout(0.2)

    def call(self, channel_features, training=False):
        out_1 = self.ff1(channel_features)
        out_1 = self.bn1(out_1, training=training)
        out_1 = self.dropout(out_1, training=training)
        out_2 = self.ff2(out_1)
        out_2 = self.bn2(out_2, training=training)
        log_sigma = self.ff3(out_2)
        log_sigma = tf.clip_by_value(log_sigma, self.min_log_sigma, self.max_log_sigma)
        # Returns: logσ²_bce, logσ²_papr, logσ²_oob, logσ²_af
        return (log_sigma[:, 0], log_sigma[:, 1], log_sigma[:, 2], log_sigma[:, 3])