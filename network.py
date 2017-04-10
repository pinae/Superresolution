#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from subpixel import PS as phase_shift
from scipy.misc import imresize
import numpy as np


class Network(object):
    def __init__(self, dimensions, batch_size, initialize_loss=True):
        self.batch_size = batch_size
        self.layer_params = []
        self.inputs = tf.placeholder(
            tf.float32, [batch_size, dimensions[1], dimensions[0], 3], name='input_images'
        ) / 255
        print("inputs shape: " + str(self.inputs.get_shape()))
        self.layer_params.append({
            'filter_count': 4*3,
            'filter_shape': [3, 3]
        })
        hidden1 = self.conv_layer("hidden1", self.layer_params[-1], self.inputs)
        print("hidden1 shape: " + str(hidden1.get_shape()))
        self.output = tf.nn.tanh(phase_shift(hidden1, 2, color=True)) * 255
        if initialize_loss:
            self.real_images = tf.placeholder(tf.float32,
                                              [self.batch_size, dimensions[1] * 2, dimensions[0] * 2, 3],
                                              name='real_images')
            self.loss = self.get_loss()
            self.summary = tf.summary.scalar("loss", self.loss)

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
        print(self.output.get_shape())
        return tf.reduce_mean(tf.square(self.real_images - self.output))

    def train_step(self, images):
        resized_images = [imresize(image, (image.shape[0] // 2, image.shape[1] // 2)) for image in images]
        train_step = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run(train_step, feed_dict={
            self.inputs: np.array(resized_images),
            self.real_images: np.array(images)
        })

    def inference(self, images):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        return sess.run(self.output, feed_dict={
            self.inputs: np.array(images)
        })
