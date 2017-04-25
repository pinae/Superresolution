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
        self.inputs = tf.placeholder(
            tf.float32, [batch_size, dimensions[1], dimensions[0], 3], name='input_images'
        )
        scaled_inputs = self.inputs / 256.0
        print("inputs shape: " + str(self.inputs.get_shape()))
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        hidden1 = self.conv_layer("hidden1", self.layer_params[-1], scaled_inputs)
        print("hidden1 shape: " + str(hidden1.get_shape()))
        # Residual layers
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_1 = self.conv_layer("hidden1", self.layer_params[-1], hidden1,
                                      weight_stddev=0.0001, bias_init=0.0) + hidden1
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_2 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_1,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_1
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_3 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_2,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_2
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_4 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_3,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_3
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_5 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_4,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_4
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_6 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_5,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_5
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_7 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_6,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_6
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_8 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_7,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_7
        self.layer_params.append({
            'filter_count': 64,
            'filter_shape': [3, 3]
        })
        res_layer_9 = self.conv_layer("hidden1", self.layer_params[-1], res_layer_8,
                                      weight_stddev=0.0001, bias_init=0.0) + res_layer_8
        # Phase shift layer
        self.layer_params.append({
            'filter_count': self.scale_factor * self.scale_factor * 3,
            'filter_shape': [3, 3]
        })
        phase_shift_layer = self.conv_layer("phase_shift_layer", self.layer_params[-1], res_layer_9)
        print("phase_shift_layer shape: " + str(phase_shift_layer.get_shape()))
        #self.output = tf.nn.tanh(phase_shift(phase_shift_layer, self.scale_factor, color=True))
        self.output = phase_shift(phase_shift_layer, self.scale_factor, color=True)
        if initialize_loss:
            self.real_images = tf.placeholder(tf.float32,
                                              [self.batch_size,
                                               dimensions[1] * self.scale_factor,
                                               dimensions[0] * self.scale_factor,
                                               3],
                                              name='real_images')
            self.loss = self.get_loss()
            self.summary = tf.summary.scalar("loss", self.loss)
            self.epoch = tf.placeholder(tf.int32, name='epoch')
            self.learning_rate = tf.train.exponential_decay(0.000001, self.epoch,
                                                            10, 0.9, staircase=True)
            self.optimized = tf.train.AdamOptimizer(self.learning_rate,
                                                    beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def initialize(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    @staticmethod
    def weight_variable(shape, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape, initial=0.1):
        initial = tf.constant(initial, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, name, params, data, weight_stddev=0.1, bias_init=0.1):
        with tf.variable_scope(name):
            weights = self.weight_variable(params['filter_shape'] + [data.get_shape().as_list()[3],
                                           params['filter_count']], stddev=weight_stddev)
            biases = self.bias_variable([params['filter_count']], initial=bias_init)
            conv = tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME')
            params['output'] = tf.nn.relu(conv + biases)
            #params['output'] = conv + biases
            params['biases'] = biases
            params['weights'] = weights
            return params['output']

    def get_loss(self):
        return tf.reduce_sum(tf.nn.l2_loss(self.output - tf.div(self.real_images, 256.0)))

    def get_batch_size(self):
        return self.batch_size

    def get_scale_factor(self):
        return self.scale_factor

    def train_step(self, images, epoch=0):
        resized_images = [imresize(image, (image.shape[0] // self.scale_factor,
                                           image.shape[1] // self.scale_factor)) for image in images]
        _, loss, lr = self.sess.run([self.optimized, self.loss, self.learning_rate], feed_dict={
            self.inputs: np.array(resized_images),
            self.real_images: np.array(images),
            self.epoch: epoch
        })
        return loss, lr

    def validation_step(self, images):
        resized_images = [imresize(image, (image.shape[0] // self.scale_factor,
                                           image.shape[1] // self.scale_factor)) for image in images]
        loss = self.sess.run([self.loss], feed_dict={
            self.inputs: np.array(resized_images),
            self.real_images: np.array(images)
        })
        return loss

    def inference(self, images):
        return self.sess.run(self.output * 256.0, feed_dict={
            self.inputs: np.array(images)
        })

    def save(self):
        save_path = self.saver.save(self.sess, os.path.abspath("network_params"))
        return save_path

    def load(self, path="network_params"):
        self.saver.restore(self.sess, os.path.abspath(path))
