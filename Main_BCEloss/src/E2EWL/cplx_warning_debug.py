import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0=all, 1=info, 2=warning, 3=error

# or also disable eager warnings by
tf.get_logger().setLevel('ERROR')

# Define two trainable variables: one real, one complex

# var = tf.Variable(tf.ones([1]), dtype=tf.float32, name='real_var')
var = tf.Variable(tf.complex(tf.ones([1]), tf.zeros([1])), dtype=tf.complex64, name='complex_var')
complex_var = tf.Variable(tf.complex(tf.ones([1]), tf.ones([1])), dtype=tf.complex64, name='complex_var')


with tf.GradientTape() as tape:
    # Combine variables in some complex operation
    combined = tf.cast(var, tf.complex64) + complex_var
    
    # Loss: sum of squared magnitude (real scalar)
    loss = tf.abs(combined)**2

grads = tape.gradient(loss, [var, complex_var])
print(grads)


