#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from subpixel import PS as phase_shift
from scipy.misc import imresize


class Network(object):
    def __init__(self, dimensions, batch_size):
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.layer_params = []
        self.inputs = tf.placeholder(tf.float32, [batch_size, dimensions[0], dimensions[1], 3], name='input_images')
        self.layer_params.append({
            'filter_count': 4*3,
            'filter_shape': [3, 3]
        })
        hidden1 = self.conv_layer("hidden1", self.layer_params[-1], self.inputs)
        print("hidden1 shape: "+str(hidden1.get_shape()))
        self.output_layer = tf.nn.tanh(phase_shift(hidden1, 2, color=True))
        self.loss = self.get_loss()
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, name, params, data):
        with tf.variable_scope(name):
            weights = self.weight_variable(params['filter_shape'] + [data.get_shape().as_list()[3],
                                           params['filter_count']])
            biases = self.bias_variable([params['filter_count']])
            conv = tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME')
            params['output'] = tf.nn.relu(conv + biases)
            params['biases'] = biases
            params['weights'] = weights
            return params['output']

    def get_loss(self):
        real_images = tf.placeholder(tf.float32, [self.batch_size, self.dimensions[0]*2, self.dimensions[1]*2, 3],
                                     name='real_images')
        print(self.output_layer.get_shape())
        loss_matrix = tf.reduce_mean(tf.square(real_images - self.output_layer))
        return tf.scalar_summary("loss", loss_matrix)

    def train_step(self, images):
        resized_images = imresize(images, (images.shape()[1] // 2, images.shape()[2] // 2))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.sess.run(train_step, feed_dict={self.input_images: resized_images, self.real_images: images})
