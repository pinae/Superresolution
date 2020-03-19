# -*- coding: utf-8 -*-
from tensorflow.compat.v1 import placeholder, truncated_normal, Session
from tensorflow.compat.v1.image import resize_bicubic
from tensorflow.compat.v1.train import AdamOptimizer, exponential_decay, Saver
import tensorflow as tf
import numpy as np
import os
tf.compat.v1.disable_eager_execution()


class Network(object):
    def __init__(self, dimensions, batch_size, initialize_loss=True, lr=0.0001, lr_stair_width=10, lr_decay=0.95):
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.scale_factor = 2
        self.layer_params = []
        self.inputs = placeholder(
            tf.float32, [batch_size, dimensions[1], dimensions[0], 3], name='input_images'
        )
        scaled_inputs = self.inputs / 256.0
        print("inputs shape: " + str(self.inputs.get_shape()))

        resized = resize_bicubic(
            scaled_inputs,
            [dimensions[1] * self.scale_factor, dimensions[0] * self.scale_factor],
            name="scale_bicubic")

        self.layer_params.append({
            'filter_count': 64 * 3,
            'filter_shape': [9, 9]
        })
        patch_extraction_layer = self.conv_layer("patch_extraction", self.layer_params[-1], resized)
        self.layer_params.append({
            'filter_count': 32 * 3,
            'filter_shape': [1, 1]
        })
        non_linear_mapping_layer = self.conv_layer("non_linear_mapping_layer", self.layer_params[-1],
                                                   patch_extraction_layer)
        self.layer_params.append({
            'filter_count': 3,
            'filter_shape': [5, 5]
        })
        self.output = self.conv_layer("reconstruction_layer", self.layer_params[-1],
                                      non_linear_mapping_layer, linear=True)

        if initialize_loss:
            self.real_images = placeholder(tf.float32,
                                           [self.batch_size,
                                            dimensions[1] * self.scale_factor,
                                            dimensions[0] * self.scale_factor,
                                            3],
                                           name='real_images')
            self.loss = self.get_loss()
            self.summary = tf.summary.scalar("loss", self.loss)
            self.epoch = placeholder(tf.int32, name='epoch')
            self.learning_rate = exponential_decay(
                lr, self.epoch, lr_stair_width, lr_decay, staircase=True)
            self.optimized = AdamOptimizer(
                self.learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        self.sess = Session()
        self.saver = Saver()

    def initialize(self):
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    @staticmethod
    def weight_variable(shape, stddev=0.1):
        initial = truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape, initial=0.1):
        initial = tf.constant(initial, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, name, params, data, weight_stddev=0.1, bias_init=0.1, linear=False):
        with tf.compat.v1.variable_scope(name):
            weights = self.weight_variable(params['filter_shape'] + [data.get_shape().as_list()[3],
                                           params['filter_count']], stddev=weight_stddev)
            biases = self.bias_variable([params['filter_count']], initial=bias_init)
            padded_data = tf.pad(data, [[0, 0],
                                        [(params['filter_shape'][0] - 1) // 2,
                                         (params['filter_shape'][0] - 1) // 2],
                                        [(params['filter_shape'][1] - 1) // 2,
                                         (params['filter_shape'][1] - 1) // 2],
                                        [0, 0]], "SYMMETRIC")
            conv = tf.nn.conv2d(padded_data, weights, strides=[1, 1, 1, 1], padding='VALID')
            if not linear:
                params['output'] = tf.nn.relu(conv + biases)
            else:
                params['output'] = conv + biases
            params['biases'] = biases
            params['weights'] = weights
            return params['output']

    def get_loss(self):
        return tf.reduce_sum(tf.nn.l2_loss(self.output - tf.raw_ops.Div(x=self.real_images, y=256.0)))

    def get_batch_size(self):
        return self.batch_size

    def get_scale_factor(self):
        return self.scale_factor

    def get_dimensions(self):
        return self.dimensions

    def train_step(self, images, target_images, epoch=0):
        _, loss, lr = self.sess.run([self.optimized, self.loss, self.learning_rate], feed_dict={
            self.inputs: np.array(images),
            self.real_images: np.array(target_images),
            self.epoch: epoch
        })
        return loss, lr

    def validation_step(self, images, target_images):
        loss = self.sess.run([self.loss], feed_dict={
            self.inputs: np.array(images),
            self.real_images: np.array(target_images)
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
