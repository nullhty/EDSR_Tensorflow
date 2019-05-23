# -*- coding: utf-8 -*-
import tensorflow as tf


def weight_variable(shape, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        weight = tf.get_variable(name, shape, initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
        return weight
    
    
def bias_variable(shape, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        bias = tf.get_variable(name, shape, initializer=tf.zeros_initializer(), dtype=tf.float32)
        return bias

    
    
def conv2d(x, shape, name, stride=[1, 1, 1, 1], pad='SAME', activation='lrelu', alpha=0.05, use_bias=True, reuse=False):

    def relu(x):
        return tf.nn.relu(x)

    def lrelu(x, alpha=0.05):
        return tf.nn.leaky_relu(x, alpha)

    w_name = name + '_w'
    b_name = name + '_b'
    weight = weight_variable(shape, w_name, reuse)
    
    y = tf.nn.conv2d(x, weight, strides=stride, padding=pad)
    if use_bias is True:
        bias = bias_variable(shape[3], b_name, reuse)
        y = y + bias
    
    if activation == 'relu':
        y = relu(y)
    elif activation == 'lrelu':
        y = lrelu(y, alpha)
    elif activation == 'None':
        y = y
    return y


def extract_feature(x, output_channel, kernel_size=[3, 3]):
    output = tf.layers.conv2d(x, output_channel, kernel_size, padding='SAME')
    return output


def residual_block(x, output_channel, kernel_size=[3, 3], block_num=32, scale=0.1):

    def residual(x, output_channel, kernel_size=[3, 3], scale=0.1):
        temp = tf.layers.conv2d(x, output_channel, kernel_size, activation=None, padding='SAME')
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, output_channel, kernel_size, activation=None, padding='SAME')
        temp *= scale
        return x + temp

    output = x
    for i in range(block_num):
        output = residual(output, output_channel, kernel_size, scale=scale)
    return output


def reconstruct(x, scale=8, features=256, activation=tf.nn.relu):

    def _phase_shift(I, r):
        return tf.depth_to_space(I, r)

    def PS(X, r, color=False):
        if color:
            Xc = tf.split(X, 3, 3)
            X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
        else:
            X = _phase_shift(X, r)
        return X

    assert scale in [2, 3, 4, 8]
    x = tf.layers.conv2d(x, features, [3, 3], activation=activation, padding='SAME')
    if scale == 2:
        ps_features = 3 * (scale ** 2)
        x = tf.layers.conv2d(x, ps_features, [3, 3], activation=activation, padding='SAME')  # Increase channel depth
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        x = tf.layers.conv2d(x, ps_features, [3, 3], activation=activation, padding='SAME')
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            x = tf.layers.conv2d(x, ps_features, [3, 3], activation=activation, padding='SAME')
            x = PS(x, 2, color=True)
    elif scale == 8:
        ps_features = 3 * (8 * 8)
        x = tf.layers.conv2d(x, ps_features, [3, 3], activation=activation, padding='SAME')
        x = PS(x, 8, color=True)
    return x


def EDSR(input_image, scale_factor=2, residual_factor=0.1):

    feature_size = 256
    residual_num = 32

    input_mean = tf.reduce_mean(input_image, 2, keepdims=True)
    input_mean = tf.reduce_mean(input_mean, 1, keepdims=True)
    input_image = input_image - input_mean

    temp = extract_feature(input_image, feature_size, kernel_size=[3, 3])
    conv_1 = temp

    temp = residual_block(temp, feature_size, kernel_size=[3, 3], block_num=residual_num, scale=residual_factor)
    temp = tf.layers.conv2d(temp, feature_size, [3, 3], padding='SAME')
    temp += conv_1

    temp = reconstruct(temp, scale_factor, feature_size, activation=None)

    output = tf.clip_by_value(temp + input_mean, 0.0, 255.0)
    return output

