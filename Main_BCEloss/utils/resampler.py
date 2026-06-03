import numpy as np
import tensorflow as tf
# from sionna.utils import register_keras_serializable

def design_fir_filter(factor, num_taps=128, window="hamming"):
    """
    Design a low-pass FIR filter for interpolation/decimation.
    factor : oversampling/decimation factor
    num_taps : filter length
    window : "hamming", "hann", or "blackman"
    """
    cutoff = 0.5 / factor   # normalized cutoff (Nyquist = 0.5)
    n = np.arange(num_taps) - (num_taps - 1) / 2
    h = np.sinc(2 * cutoff * n)

    if window == "hamming":
        w = np.hamming(num_taps)
    elif window == "hann":
        w = np.hanning(num_taps)
    elif window == "blackman":
        w = np.blackman(num_taps)
    else:
        raise ValueError("Unknown window type")

    h = h * w
    h = h / np.sum(h)  # normalize DC gain
    return tf.constant(h, dtype=tf.float32)


def conv1d_complex(x, fir_filter):
    """
    Apply real-valued FIR filter to complex input along last axis.
    x: [..., N] complex64
    fir_filter: [L] real32
    returns: [..., N] complex64
    """
    # Expand to [batch, length, 1]
    x_real = tf.expand_dims(tf.math.real(x), axis=-1)
    x_imag = tf.expand_dims(tf.math.imag(x), axis=-1)

    # conv1d requires [batch, length, channels]
    x_real = tf.nn.conv1d(x_real, fir_filter[:, None, None], stride=1, padding="SAME")
    x_imag = tf.nn.conv1d(x_imag, fir_filter[:, None, None], stride=1, padding="SAME")

    # Remove channel dimension
    x_real = tf.squeeze(x_real, axis=-1)
    x_imag = tf.squeeze(x_imag, axis=-1)

    return tf.complex(x_real, x_imag)


# @register_keras_serializable(package="CustomSionna")
class Interpolator(tf.keras.layers.Layer):
    def __init__(self, factor, fir_filter=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        if fir_filter is None:
            fir_filter = design_fir_filter(factor)
        self.fir_filter = fir_filter

    def call(self, x):
        # x: [..., N] complex64
        # Insert zeros (upsample)
        x = tf.expand_dims(x, -1)  # [..., N, 1]
        rank = len(x.shape)
        paddings = [[0,0]]*(rank-2) + [[0,0], [0, self.factor-1]]
        x = tf.pad(x, paddings)    # [..., N, factor]
        shape = tf.shape(x)
        x = tf.reshape(x, tf.concat([shape[:-2], [shape[-2]*shape[-1]]], axis=0))  # [..., N*factor]

        # Apply FIR filter separately on real & imag
        x = conv1d_complex(x, self.fir_filter)
        return x * self.factor

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


# @register_keras_serializable(package="CustomSionna")
class Decimator(tf.keras.layers.Layer):
    def __init__(self, factor, fir_filter=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        if fir_filter is None:
            fir_filter = design_fir_filter(factor)
        self.fir_filter = fir_filter

    def call(self, x):
        # x: [..., N] complex64
        # Filter
        x = conv1d_complex(x, self.fir_filter)
        # Downsample
        return x[..., ::self.factor]

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config
if __name__ == "__main__":

    BatchSize = 32

    interp = Interpolator(factor=4)   # 4x interpolation
    decim  = Decimator(factor=4)      # 4x decimation

    x = tf.ones([BatchSize,50])  # example input

    x_up = interp(x)
    x_down = decim(x_up)
    x=1