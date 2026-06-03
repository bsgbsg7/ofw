import tensorflow as tf

class Receive_coeffs(tf.keras.layers.Layer):
    def __init__(self, shape, initializer="random_normal", **kwargs):
        super().__init__(dtype=tf.complex64, **kwargs)
        self.shape = shape
        self.real_initializer = tf.keras.initializers.RandomNormal(seed=42)
        self.imag_initializer = tf.keras.initializers.RandomNormal(seed=84)

    def build(self, input_shape=None):

        # Create two real-valued initializations (for real and imaginary parts)
        real_init = self.real_initializer(shape=self.shape, dtype=tf.float32)
        imag_init = self.imag_initializer(shape=self.shape, dtype=tf.float32)
        complex_init = tf.complex(real_init, imag_init)

        self.Receive_coeffs_var = self.add_weight(
            shape=self.shape,
            dtype=tf.complex64,
            initializer=lambda shape, dtype=None: complex_init ,
            trainable=True,
            name="Receive_filter_coeffs"
        )

    def call(self, inputs=None):
        return self.Receive_coeffs_var
    

class Transmit_coeffs(tf.keras.layers.Layer):
    def __init__(self, shape, initializer="random_normal", **kwargs):
        super().__init__(dtype=tf.complex64, **kwargs)
        self.shape = shape
        self.real_initializer = tf.keras.initializers.RandomNormal(seed=42)
        self.imag_initializer = tf.keras.initializers.RandomNormal(seed=84)

    def build(self, input_shape=None):

        # Create two real-valued initializations (for real and imaginary parts)
        real_init = self.real_initializer(shape=self.shape, dtype=tf.float32)
        imag_init = self.imag_initializer(shape=self.shape, dtype=tf.float32)
        complex_init = tf.complex(real_init, imag_init)

        self.Transmit_coeffs_var = self.add_weight(
            shape=self.shape,
            dtype=tf.complex64,
            initializer=lambda shape, dtype=None: complex_init ,
            trainable=True,
            name="Transmit_filter_coeffs"
        )

    def call(self, inputs=None):
        return  self.Transmit_coeffs_var