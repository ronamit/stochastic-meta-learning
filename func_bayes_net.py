
from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import parameters as prm
import common as cmn
from common import subset_with_substring, get_var_from_list, get_var_with_substring

#-----------------------------------------------------------------------------------------------------------#
#   Extract parameters
#-----------------------------------------------------------------------------------------------------------#


bias_init = prm.bias_init   # Add some positive bias in the initialization to avoid 'dead' ReLu neurons
sigma_prior_init = prm.sigma_prior_init

# -----------------------------------------------------------------------------------------------------------#
#  Stochastic Layer
# -----------------------------------------------------------------------------------------------------------#


def stochastic_linear_layer(x, net_name, init_source, layer_name, weight_shape, bias_shape, eps_std):

    with tf.variable_scope(layer_name,
                           initializer=tf.random_normal_initializer(0, prm.random_init_std)):  # Default initializer

        # Variables initialization:
        w_mu, b_mu, w_log_sigma, b_log_sigma = init_layer(layer_name, init_source, weight_shape, bias_shape)

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


def init_layer(layer_name, init_source, weight_shape, bias_shape):
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
                               initializer=get_var_with_substring(source_collection, layer_name+'/w_mu').initialized_value())
        b_mu = tf.get_variable('b_mu',
                               initializer=get_var_with_substring(source_collection, layer_name+'/b_mu').initialized_value())
        w_log_sigma = tf.get_variable('w_log_sigma',
                                      initializer=get_var_with_substring(source_collection, layer_name+'/w_log_sigma').initialized_value())
        b_log_sigma = tf.get_variable('b_log_sigma',
                                      initializer=get_var_with_substring(source_collection, layer_name+'/b_log_sigma').initialized_value())
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

    # Get net parameters:
    input_size = prm.input_size
    n_labels = prm.n_labels
    width_per_layer = prm.width_per_layer

    n_layers = len(width_per_layer)  # number of hidden layers
    prev_dim = input_size
    h = input  # input

    # hidden layers:
    for i_layer in range(n_layers):
        layer_name = 'hidden_layer_' + str(i_layer)
        new_dim = width_per_layer[i_layer]

        # Fully-connected layer:
        h = stochastic_linear_layer(h, net_name, init_source, layer_name=layer_name,
                                    weight_shape=[prev_dim, new_dim], bias_shape=[new_dim], eps_std=eps_std)
        # activation function:
        h = tf.nn.elu(h)

        prev_dim = new_dim

    # output layer:
    layer_name = 'out_layer'
    net_out = stochastic_linear_layer(h, net_name, init_source, layer_name=layer_name,
                                      weight_shape=[prev_dim, n_labels], bias_shape=[n_labels], eps_std=eps_std)

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
        curr_kl_dist = tf.nn.relu(curr_kl_dist) # To avoid negative KL TODO: Find better fix
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



def single_task_complexity(objective_type, n_samples, kl_dist):

    if objective_type == 'PAC_Bayes_McAllaster':
        delta = 0.95
        complex_term = tf.sqrt((1 / (2 * n_samples)) * (kl_dist + np.log(2*np.sqrt(n_samples) / delta))) - \
                       np.sqrt((1 / (2 * n_samples)) * (np.log(2*np.sqrt(n_samples) / delta)))
        # I subtracted a const so that the optimization could reach 0

    elif objective_type == 'PAC_Bayes_Pentina':
        complex_term = np.sqrt(1 / n_samples) * kl_dist

    elif objective_type == 'Variational_Bayes':
        # Since we approximate the expectation of the likelihood of all samples,
        # we need to multiply by the average_loss by total number of samples
        # Then we normalize the objective by n_samples
        complex_term = (1 / n_samples) * kl_dist

    elif objective_type == 'Bayes_No_Prior':
        complex_term = 0

    else:
        raise ValueError('Invalid objective_type')

    return complex_term



def single_task_objective(objective_type, average_loss, n_samples, kl_dist):

    if objective_type == 'PAC_Bayes_Seeger':
        # Seeger complexity is different since it requires the average_loss
        p = 1e-9  # add small positive number to avoid sqrt of negative number due to numerical errors
        delta = 0.95
        seeger_eps = (1 / n_samples) * (kl_dist + np.log(2*np.sqrt(n_samples) / delta) )
        objective = average_loss + 2 * seeger_eps + tf.sqrt(2 * seeger_eps * average_loss + p)

    else:
        objective = average_loss + single_task_complexity(objective_type, n_samples, kl_dist)

    return objective
