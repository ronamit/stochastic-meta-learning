

from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np

import tensorflow as tf

import parameters as prm


def DoDropOut(a, dropout_flag):
    if dropout_flag:
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


def network_model(x, dropout_flag):
    """network_model builds the graph for a deep net for classifying digits."""

    # Get net parameters:
    input_size = prm.input_size
    n_labels = prm.n_labels
    width_per_layer = prm.width_per_layer

    n_layers = len(width_per_layer) # number of hidden layers
    prev_dim = input_size
    h = x # input

    # hidden layers:
    for i_layer in range(n_layers):
        layer_name = 'hidden_layer' + str(i_layer)
        new_dim = width_per_layer[i_layer]

        # Fully-connected layer:
        h = deterministic_linear_layer(h, layer_name=layer_name,
                                       weight_shape=[prev_dim, new_dim], bias_shape=[new_dim])
        # activation function:
        h = tf.nn.elu(h)
        h = DoDropOut(h, dropout_flag)
        prev_dim = new_dim

    # output layer:
    layer_name = 'out_layer'
    net_out = deterministic_linear_layer(h, layer_name=layer_name, weight_shape=[prev_dim, n_labels],
                                      bias_shape=[n_labels])

    return net_out

