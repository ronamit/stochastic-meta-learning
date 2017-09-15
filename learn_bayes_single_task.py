
from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import parameters as prm
import common as cmn
import func_bayes_net as sf

#-----------------------------------------------------------------------------------------------------------#
#   global parameters
#-----------------------------------------------------------------------------------------------------------#
input_size = prm.input_size
n_labels = prm.n_labels

# -----------------------------------------------------------------------------------------------------------#
#  Single Task Bayesian Learning Function
# -----------------------------------------------------------------------------------------------------------#


def learn_task(data, objective_type, n_steps = prm.default_n_steps):

    n_samples = data.train.num_examples
    print('Number training of samples: ', n_samples)

    # -----------------------------------------------------------------------------------------------------------#
    #   Define the model's graph
    # -----------------------------------------------------------------------------------------------------------#

    graph = tf.Graph()

    with graph.as_default():
        with tf.variable_scope("graph"):
            
            #  The input: (batch of samples)
            x = tf.placeholder(tf.float32, [None, input_size])

            #  Ground truth labels:
            labels = tf.placeholder(tf.float32, [None, n_labels])

            # STD of epsilon for re-parametrization trick:
            eps_std = tf.placeholder(tf.float32, [])         

            # Initialize the weight's prior variables:
            with tf.variable_scope('prior'):
                # call network with dummy inputs to add prior variables to graph
                sf.network_model('prior', init_source='constants', input=x, eps_std=1)

            # The net with the posterior variables acts on the inputs:
            with tf.variable_scope('posterior'):
                net_out = sf.network_model('posterior', init_source='random', input=x, eps_std=eps_std)

            # The empirical loss:
            average_loss = tf.reduce_mean(cmn.loss_function(labels, net_out))

            # The Kullback-Leibler divergence between posterior and prior:
            kl_dist = sf.calculate_kl_dist('posterior', 'prior')

            objective = sf.single_task_objective(objective_type, average_loss, n_samples, kl_dist)

            # Learning rate:
            learning_rate = prm.learning_rate

            # Optimization step:
            posterior_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='graph/posterior')
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(objective, var_list=posterior_trainable_vars)

            # Compare net output to true labels:
            correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.reset_default_graph()

    # -----------------------------------------------------------------------------------------------------------#
    #  Run Learning
    # -----------------------------------------------------------------------------------------------------------#


    n_steps_stage_1 = int(n_steps * prm.steps_stage_1_ratio)
    n_steps_stage_2 = n_steps - n_steps_stage_1
    n_steps_with_full_eps_std = int(n_steps_stage_2 * prm.steps_with_full_eps_ratio)

    with tf.Session(graph=graph) as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())

        for i_step in xrange(n_steps):

            # We gradually increase epsilon's STD from 0 to 1.
            # The reason is that using 1 from the start results in high variance gradients.
            if i_step >= n_steps_stage_1:
                cur_eps_std = prm.epsilonStdDefault * (i_step - n_steps_stage_1) / (n_steps_stage_2 - n_steps_with_full_eps_std)
            else:
                cur_eps_std = 0.0
            cur_eps_std = min(max(cur_eps_std, 0.0), 1.0)  # keep in [0,1]

            # Draw random mini-batch:
            batch = data.train.next_batch(prm.batch_size, shuffle=True)

            feed_dict = {x: batch[0], labels: batch[1], eps_std: cur_eps_std}

            # Take gradient step:
            train_step.run(feed_dict=feed_dict)

            #  training performance evaluation:
            if i_step % 5000 == 0:
                (cur_accuracy, cur_objective, cur_kl_dist, cur_average_loss) = \
                    sess.run([accuracy, objective, kl_dist, average_loss], feed_dict=feed_dict)

                print('step %d, eps: %g, accuracy %g, objective: %g, ' % (i_step, cur_eps_std, cur_accuracy, cur_objective) +
                      'kl dist: %g, avg loss: %g' % (cur_kl_dist, cur_average_loss) )

        # Evaluate on test set:
        # Evaluate on test with epsilon = 0 (maximum of posterior)
        # TODO: Sample different outputs and take majority vote
        feed_dict = {x: data.test.images, labels: data.test.labels, eps_std: 0}
        test_accuracy = sess.run(accuracy, feed_dict=feed_dict)

        print('test accuracy %g' % test_accuracy)

    return test_accuracy

