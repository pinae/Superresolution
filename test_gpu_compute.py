# -*- coding: utf-8 -*-
import tensorflow as tf


# Creates a graph.
@tf.function
def matmul(x, y):
    return tf.matmul(x, y)


with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    # Runs the op.
    print(matmul(a, b))
