
"""
This module defines the split-step Fourier method to approximate the solution of
the nonlinear Schroedinger equation.
"""

import tensorflow as tf
from sionna.phy import config, constants
from sionna.phy import Block
from sionna.phy.channel import utils

class PA(Block):
    # pylint: disable=line-too-long
    r"""
    Block implementing the split-step Fourier method (SSFM)


    Running:

    >>> # x is the optical input signal
    >>> y = ssfm(x)

    Parameters
    ----------


    with_amplification : `bool`, (default `False`)
        Enable ideal inline amplification and corresponding
        noise

    with_attenuation : `bool`, (default `True`)
        Enable attenuation


    with_nonlinearity : `bool`, (default `True`)
        Apply Kerr nonlinearity

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,n] or [...,2,n], `tf.complex`
        Input signal in :math:`(\sqrt{\text{W}})`. If ``with_manakov``
        is `True`, the second last dimension is interpreted
        as x- and y-polarization, respectively.

    Output
    ------
    y : Tensor (same shape as ``x``), `tf.complex`
        Channel output
    """
    def __init__(self,
                 a=1,
                 b=0,
                 c=0,
                 with_amplification=False,
                 with_attenuation=False,
                 with_nonlinearity=True,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        self._a = tf.cast(a, dtype=self.cdtype)
        self._b = tf.cast(b, dtype=self.cdtype)
        self._c = tf.cast(c, dtype=self.cdtype)

    def call(self, x):
        output = self._a * x + self._b * tf.pow(x, 2) + self._c * tf.pow(x, 3)
        return output

if __name__ == "__main__":
    pa = PA(a=2)
    x = tf.complex(2.0, 3.0)                # shape: () — scalar complex
    x = tf.fill([32, 1], x) 
    z = pa(x)
    s = None
