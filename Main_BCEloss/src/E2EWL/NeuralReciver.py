from keras import layers
from keras.layers import Dense, Layer
from config import NUM_BITS_PER_SYMBOL
import tensorflow as tf

class E2EWLNeuralReciver(Layer): # Inherits from Keras Layer

    def __init__(self, K, Np, Ns, is_multypath=True):
        super().__init__()

        # general Params
        self.is_multypath = is_multypath  # AWGN model only
        self.Np = Np                # number of pilots for non-AWGN case
        self.Ns = Ns                # number of pilots layer output for non-AWGN case

        # Layers
        if self.is_multypath:
            self.pilot_layer = PilotLayer(Ns)
        self.initial_conv = layers.Conv1D(256, 3, padding="same", dilation_rate=1)
        self.resnet1 = ResNetBlock1D(256, 3, dilation_rate=3)
        self.resnet2 = ResNetBlock1D(256, 3, dilation_rate=6)
        self.resnet3 = ResNetBlock1D(256, 3, dilation_rate=6)
        self.resnet4 = ResNetBlock1D(256, 3, dilation_rate=3)
        self.final_conv = layers.Conv1D(K, 1, padding="same", dilation_rate=1)

    def call(self, r, training=False):
        
        c2r_output = tf.stack([tf.math.real(r), tf.math.imag(r)], axis=-1)
        
        if self.is_multypath:
            pilot_input = c2r_output[:, :self.Np, :]                    
            signal_input = c2r_output[:, self.Np:, :]                    
            pilot_features  = self.pilot_layer(pilot_input, training=training)
            pilot_features = tf.expand_dims(pilot_features, axis=1)
            pilot_features = tf.tile(pilot_features, [1, tf.shape(signal_input)[1], 1]) # repeat along symbols dim
            x = tf.concat([signal_input, pilot_features], axis=-1)                      # concat along channels dim
        else:
            x = c2r_output
        
        x = self.initial_conv(x)
        x = self.resnet1(x, training=training)
        x = self.resnet2(x, training=training)
        x = self.resnet3(x, training=training)
        x = self.resnet4(x, training=training)
        x = self.final_conv(x)
        llr = tf.reshape(x, [tf.shape(r)[0], -1]) 
        return llr
    
class PilotLayer(tf.keras.layers.Layer):
    def __init__(self, Ns, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(256, 'relu')
        self.dense2 = Dense(256, 'relu')
        self.dense3 = Dense(Ns,  'linear')

    def call(self, x, training=False):
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = self.dense1(x)
        x = self.dense2(x)
        z = self.dense3(x)
        return z
    
class ResNetBlock1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.sepconv1 = layers.SeparableConv1D(filters, kernel_size, 
                                               padding="same", 
                                               dilation_rate=dilation_rate)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.sepconv2 = layers.SeparableConv1D(filters, kernel_size, 
                                               padding="same", 
                                               dilation_rate=dilation_rate)

    def call(self, x, training=False):
        shortcut = x 
        out = self.bn1(x, training=training)
        out = self.relu1(out)
        out = self.sepconv1(out)

        out = self.bn2(out, training=training)
        out = self.relu2(out)
        out = self.sepconv2(out)

        return out + shortcut