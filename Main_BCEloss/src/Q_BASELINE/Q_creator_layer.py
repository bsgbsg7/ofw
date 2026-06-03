import tensorflow as tf
import numpy as np
from keras.layers import Dense, Layer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.General_helpers import idft_matrix


class Q_creator_layer(Layer):
    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)
        self.IDFT_matrix = idft_matrix(N)
        # self.dense1 = Dense(256, 'relu')
        # self.dense2 = Dense(256, 'relu')
        # self.dense3 = Dense(Ns,  'linear')

    def call(self, h, training=False):
        # x = tf.reshape(x, [tf.shape(x)[0], -1])
        # x = self.dense1(x)
        # x = self.dense2(x)
        # z = self.dense3(x)
        return self.IDFT_matrix#z
    

if __name__ == "__main__":
    N = 4
    F = idft_matrix(N)
    print(F)