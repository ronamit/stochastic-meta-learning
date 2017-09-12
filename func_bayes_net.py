
from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import parameters as prm
import common as cmn
from common import subset_with_substring, get_var_from_list

#-----------------------------------------------------------------------------------------------------------#
#   Extract parameters
#-----------------------------------------------------------------------------------------------------------#
input_size = prm.input_size
n_hidden1 = prm.n_hidden1
n_hidden2 = prm.n_hidden2
n_labels = prm.n_labels

bias_init = prm.bias_init   # Add some positive bias in the initialization to avoid 'dead' ReLu neurons
sigma_prior_init = prm.sigma_prior_init

# -----------------------------------------------------------------------------------------------------------#
#  Stochastic Layer
# -----------------------------------------------------------------------------------------------------------#


def stochastic_linear_layer(x, net_name, init_source, layer_name, weight_shape, bias_shape, eps_std):

    with tf.variable_scope(layer_name,
                           initializer=tf.random_normal_initializer(0, prm.random_init_std)):  # Default initializer

        # Variables initialization:
        w_mu, b_mu, w_log_sigma, b_log_sigma = init_layer(init_source, weight_shape, bias_shape)

        # Transform from log-sigma to sigma (ensures positive sigma):
        w_sigma_sqr = tf.exp(2*w_log_sigma, 'w_sigma_sqr')
        b_sigma_sqr = tf.exp(2*b_log_sigma, 'b_sigma_sqr')

        # Add variables to collection:
        cmn.add_tensors_to_collection(net_name, [w_mu, b_mu, w_sigma_sqr, b_sigma_sqr, w_log_sigma, b_log_sigma])

        # Layer computations (based on "Variational Dropout and the Local Reparameterization Trick", Kingma et.al 2015)
        out_mean = tf.matmul(x, w_mu) + b_mu
        if eps_std == 0.0:
            layer_out = out_mean
        else:
            out_var = tf.matmul(tf.square(x), w_sigma_sqr) + b_sigma_sqr
            ksi = tf.random_normal(tf.shape(out_mean), mean=0.0, stddev=eps_std)
            layer_out = out_mean + tf.multiply(tf.sqrt(out_var), ksi)

        return layer_out


def init_layer(init_source, weight_shape, bias_shape):
    # Variables initialization:
    if init_source == 'constants':
        # Initial prior variables by constants assignment
        w_mu = tf.get_variable('w_mu',
                               shape=weight_shape, initializer=tf.constant_initializer(0))
        w_log_sigma = tf.get_variable('w_log_sigma',
                                      shape=weight_shape, initializer=tf.constant_initializer(np.log(sigma_prior_init)))
        b_mu = tf.get_variable('b_mu',
                               shape=bias_shape, initializer=tf.constant_initializer(bias_init))
        b_log_sigma = tf.get_variable('b_log_sigma',
                                      shape=bias_shape, initializer=tf.constant_initializer(np.log(sigma_prior_init)))
    elif init_source == 'random':
        # Initialize variables randomly
        w_mu = tf.get_variable('w_mu', shape=weight_shape)
        b_mu = tf.get_variable('b_mu', shape=bias_shape,
                               initializer=tf.random_normal_initializer(bias_init, prm.random_init_std))
        w_log_sigma = tf.get_variable('w_log_sigma', shape=weight_shape)
        b_log_sigma = tf.get_variable('b_log_sigma', shape=bias_shape)
    else:
        # In this case "init_source" is the name of a collection of tensors that are used for initialization
        source_collection = tf.get_collection(init_source)
        if not source_collection:
            raise ValueError('Invalid init_source')
        w_mu = tf.get_variable('w_mu',
                               shape=weight_shape, initializer=get_var_from_list(source_collection, 'w_mu'))
        b_mu = tf.get_variable('w_mu',
                               shape=weight_shape, initializer=get_var_from_list(source_collection, 'b_mu'))
        w_log_sigma = tf.get_variable('w_mu',
                                      shape=weight_shape,
                                      initializer=get_var_from_list(source_collection, 'w_log_sigma'))
        b_log_sigma = tf.get_variable('w_mu',
                                      shape=weight_shape,
                                      initializer=get_var_from_list(source_collection, 'b_log_sigma'))
    return w_mu, b_mu, w_log_sigma, b_log_sigma
# -----------------------------------------------------------------------------------------------------------#
#  Stochastic Network Model
# -----------------------------------------------------------------------------------------------------------#


def network_model(net_name, init_source, input, eps_std):

    """network_model builds the graph for a deep net for classification
    Args:
      x: an input tensor with the dimensions (N_examples, input_size)
    Returns:
      y:  y is a tensor of shape (N_examples, n_labels), with values
      equal to the logits of classifying the input into one of n_labels classes
    """

    # Fully connected layer 1
    a1 = stochastic_linear_layer(input, net_name, init_source, "layer1", weight_shape=[input_size, n_hidden1],
                                 bias_shape=[n_hidden1], eps_std=eps_std)
    h1 = tf.nn.elu(a1)

    # Fully connected layer 2
    a2 = stochastic_linear_layer(h1, net_name, init_source, "layer2", weight_shape=[n_hidden1, n_hidden2],
                                 bias_shape=[n_hidden2], eps_std=eps_std)
    h2 = tf.nn.elu(a2)

    #  Fully connected layer 3 - Map the  features to 10 classes, one for each digit
    net_out = stochastic_linear_layer(h2, net_name, init_source, "layer3", weight_shape=[n_hidden2, n_labels],
                                      bias_shape=[n_labels], eps_std=eps_std)

    return net_out


def calculate_kl_dist(posterior_collection, prior_collection):
    '''
    Calculate the KL (Kullback-Leibler) divergence between
    the two factorized Gaussian  distributions
    '''

    # Get the current parameters of the prior and posterior distributions:
    posterior_tensors = tf.get_collection(posterior_collection)
    prior_tensors = tf.get_collection(prior_collection)

    post_mu_list = subset_with_substring(posterior_tensors, '_mu:')
    prior_mu_list = subset_with_substring(prior_tensors, '_mu:')
    post_sigma_sqr_list = subset_with_substring(posterior_tensors, '_sigma_sqr:')
    prior_sigma_sqr_list = subset_with_substring(prior_tensors, '_sigma_sqr:')
    post_log_sigma_list = subset_with_substring(posterior_tensors, '_log_sigma:')
    prior_log_sigma_list = subset_with_substring(prior_tensors, '_log_sigma:')

    # Calculate KL distance:
    kl_dist = 0
    for ii, _ in enumerate(post_mu_list):
        mu_post = post_mu_list[ii]
        mu_prior = prior_mu_list[ii]
        sigma_sqr_post = post_sigma_sqr_list[ii]
        sigma_sqr_prior = prior_sigma_sqr_list[ii]
        log_sigma_post = post_log_sigma_list[ii]
        log_sigma_prior = prior_log_sigma_list[ii]

        # Calculate the contribution of current weights to the KL term:
        p = 1e-9  # add small positive number to avoid division by zero due to numerical errors

        curr_kl_dist = tf.reduce_sum(log_sigma_prior - log_sigma_post +
                                     tf.divide(tf.square(mu_post - mu_prior) + sigma_sqr_post,
                                               2 * sigma_sqr_prior + p)) - 0.5

        # debug assertion: KL must be positive:
        with tf.control_dependencies([tf.assert_positive(curr_kl_dist + p)]):
            kl_dist += curr_kl_dist

    return kl_dist


def calc_param_norm(prior_collection):
    ''' Calculate the total L2 norm of the parameters of a distribution (to be used by hyper-prior) '''

    # Get the current parameters of the prior distribution:
    prior_vars = tf.get_collection(prior_collection)
    prior_mu_vars = [var for var in prior_vars if '_mu:' in var.name]
    prior_sigma_sqr_vars = [var for var in prior_vars if '_sigma_sqr:' in var.name]

    # Calculate KL distance:
    total_norm = 0
    for ii, _ in enumerate(prior_mu_vars):
        mu_prior = prior_mu_vars[ii]
        sigma_sqr_prior = prior_sigma_sqr_vars[ii]

        # Calculate the contribution of current weights to the KL term:
        total_norm += tf.reduce_sum(tf.square(mu_prior)) + tf.reduce_sum(sigma_sqr_prior)

    return total_norm


def single_task_objective(objective_type, average_loss, n_samples, kl_dist):

    if objective_type == 'PAC_Bayes_McAllaster':
        delta = 0.95
        # log_inv_delta = tf.cast(np.log(1 / delta), tf.float32)
        objective = average_loss + tf.sqrt((1 / (2 * n_samples)) * (kl_dist + np.log(2*np.sqrt(n_samples) / delta)))

    elif objective_type == 'PAC_Bayes_Pentina':
        objective = average_loss + np.sqrt(1 / n_samples) * kl_dist

    elif objective_type == 'PAC_Bayes_Seeger':
        delta = 0.95
        seeger_eps = (1 / n_samples) * (kl_dist + np.log(2*np.sqrt(n_samples) / delta))
        objective = average_loss + 2 * seeger_eps + tf.sqrt(2 * seeger_eps * average_loss)

    elif objective_type == 'Variational_Bayes':
        # Since we approximate the expectation of the likelihood of all samples,
        # we need to multiply by the average_loss by total number of samples
        # Then we normalize the objective by n_samples
        objective = average_loss + (1 / n_samples) * kl_dist

    elif objective_type == 'Bayes_No_Prior':
        objective = average_loss

    else:
        raise ValueError('Invalid objective_type')

    return objective
