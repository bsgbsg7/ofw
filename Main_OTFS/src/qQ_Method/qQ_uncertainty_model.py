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