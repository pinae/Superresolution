#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from subpixel import PS as phase_shift
from scipy.misc import imresize
import numpy as np
import os


class Network(object):
    def __init__(self, dimensions, batch_size, initialize_loss=True):
        self.batch_size = batch_size
        self.scale_factor = 2
        self.layer_params = []
        self.inputs = tf.div(tf.placeholder(
            tf.float32, [batch_size, dimensions[1], dimensions[0], 3], name='input_images'
        ), 256.0)
        print("inputs shape: " + str(self.inputs.get_shape()))
        self.layer_params.append({
            'filter_count': 3,
            'filter_shape': [1, 1]
        })
        hidden1 = self.conv_layer("hidden1", self.layer_params[-1], self.inputs)
        print("hidden1 shape: " + str(hidden1.get_shape()))
        self.layer_params.append({
            'filter_count': self.scale_factor * self.scale_factor * 3,
            'filter_shape': [3, 3]
        })
        hidden2 = self.conv_layer("hidden2", self.layer_params[-1], hidden1)
        print("hidden2 shape: " + str(hidden2.get_shape()))
        #self.output = tf.nn.tanh(phase_shift(hidden2, self.scale_factor, color=True)) * 255
        self.output = phase_shift(hidden2, self.scale_factor, color=True) * 255
        #self.output = hidden2
        if initialize_loss:
            self.real_images = tf.placeholder(tf.float32,
                                              [self.batch_size,
                                               dimensions[1] * self.scale_factor,
                                               dimensions[0] * self.scale_factor,
                                               3],
                                              name='real_images')
            self.loss = self.get_loss()
            self.summary = tf.summary.scalar("loss", self.loss)
            self.optimized = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def initialize(self):
        init = tf.global_variables_initializer()
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
            #params['output'] = tf.nn.relu(conv + biases)
            params['output'] = conv + biases
            params['biases'] = biases
            params['weights'] = weights
            return params['output']

    def get_loss(self):
        return tf.reduce_mean(tf.square(self.output - tf.div(self.real_images, 256.0)))

    def get_batch_size(self):
        return self.batch_size

    def get_scale_factor(self):
        return self.scale_factor

    def train_step(self, images):
        resized_images = [imresize(image, (image.shape[0] // self.scale_factor,
                                           image.shape[1] // self.scale_factor)) for image in images]
        self.sess.run(self.optimized, feed_dict={
            self.inputs: np.array(resized_images),
            self.real_images: np.array(images)
        })

    def inference(self, images):
        return self.sess.run(self.output * 256, feed_dict={
            self.inputs: np.array(images)
        })

    def save(self):
        save_path = self.saver.save(self.sess, os.path.abspath("network_params"))
        return save_path

    def load(self, path="network_params"):
        self.saver.restore(self.sess, os.path.abspath(path))
