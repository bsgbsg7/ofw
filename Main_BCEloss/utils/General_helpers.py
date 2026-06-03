import numpy as np
import tensorflow as tf

# generate DFT matrix
def dft_matrix(N, dtype=tf.complex128):
    n = tf.range(N, dtype=tf.float64)             # [0, 1, ..., N-1]
    k = tf.reshape(n, (-1, 1))                    # column vector
    # Exponent = +2π * k * n / N
    exponent = - 2.0 * np.pi * k * n / N
    exponent = tf.cast(exponent, tf.complex128)
    exponent = 1j * exponent
    W_inv = tf.exp(exponent) / tf.cast(tf.sqrt(tf.cast(N,tf.float64)), dtype)  # scale by 1/N
    return tf.cast(W_inv, dtype)

# generate IDFT matrix
def idft_matrix(N, dtype=tf.complex128):
    n = tf.range(N, dtype=tf.float64)             # [0, 1, ..., N-1]
    k = tf.reshape(n, (-1, 1))                    # column vector
    # Exponent = +2π * k * n / N
    exponent = 2.0 * np.pi * k * n / N
    exponent = tf.cast(exponent, tf.complex128)
    exponent = 1j * exponent
    W_inv = tf.exp(exponent) / tf.cast(tf.sqrt(tf.cast(N,tf.float64)), dtype)  # scale by 1/N
    return tf.cast(W_inv, dtype)

def make_shift_P(n, dtype=tf.complex64):
    # Right cyclic shift: (P x)[i] = x[(i-1) mod n]
    # We'll create P such that P @ e_j = e_{(j+1) mod n}
    P = tf.zeros([n, n], dtype=dtype)
    idx = tf.range(n)
    P = tf.tensor_scatter_nd_update(
        P,
        indices=tf.stack([idx, (idx - 1) % n], axis=1),
        updates=tf.ones([n], dtype=dtype)
    )
    return P

if __name__ == "__main__":
    
    # produce normalized IFFT matrix
    N = 16
    F = idft_matrix(N)
    
    # pick a colum
    k=3
    f_k = tf.expand_dims(F[:,k], axis=1)
    
    # shift it cyclic by 2
    P_1 = make_shift_P(N, dtype=f_k.dtype)
    f_k_shifted = P_1 @ P_1 @ f_k

    # varify lemma 1 by making sure is_zero is a zero vector
    is_zero = f_k * ( 1 / ( tf.cast(tf.sqrt(tf.cast(N,f_k.dtype)), f_k.dtype)*f_k[2,0] ) ) - f_k_shifted