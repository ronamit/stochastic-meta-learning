

from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np

import tensorflow as tf

import parameters as prm

input_size = prm.input_size
n_hidden1 = prm.n_hidden1
n_hidden2 = prm.n_hidden2
n_labels = prm.n_labels


def DoDropOut(a, dropoutFlag):
    if dropoutFlag:
        return tf.nn.dropout(a, keep_prob=0.5)
    else:
        return a

# -----------------------------------------------------------------------------------------------------------#
#  Deterministic Network
# -----------------------------------------------------------------------------------------------------------#
def deterministic_linear_layer(x, layer_name, weight_shape, bias_shape):
    with tf.variable_scope(layer_name,
                           initializer=tf.random_normal_initializer(0, prm.random_init_std),  # Default initializer
                           ):

        w = tf.get_variable("weights", shape=weight_shape)
        b = tf.get_variable("biases", shape=bias_shape,
                            initializer=tf.constant_initializer(
                                prm.bias_init))  # Some positive bias in the initialization to avoid "dead" ReLu neurons

    output = tf.matmul(x, w) + b
    return output


def network_model(x, dropoutFlag):
    """network_model builds the graph for a deep net for classifying digits."""

    with tf.variable_scope('net'):
        # Fully connected layer 1
        h_fc1 = tf.nn.elu(deterministic_linear_layer(x, layer_name="layer1", weight_shape=[input_size, n_hidden1],
                                                     bias_shape=[n_hidden1]))
        h_fc1 = DoDropOut(h_fc1, dropoutFlag)

        # Fully connected layer 2
        h_fc2 = tf.nn.elu(deterministic_linear_layer(h_fc1, layer_name="layer2", weight_shape=[n_hidden1, n_hidden2],
                                                     bias_shape=[n_hidden2]))
        h_fc2 = DoDropOut(h_fc2, dropoutFlag)

        #  Fully connected layer 3 - Map the  features to 10 classes, one for each digit
        yOut = deterministic_linear_layer(h_fc2, layer_name="layer3", weight_shape=[n_hidden2, n_labels],
                                          bias_shape=[n_labels])
    return yOut

