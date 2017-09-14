


from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

import tensorflow as tf

import parameters as prm
import func_standard_net

input_size = prm.input_size
n_hidden1 = prm.n_hidden1
n_hidden2 = prm.n_hidden2
n_labels = prm.n_labels


# -----------------------------------------------------------------------------------------------------------#
#  Single Task Deterministic Learning Function
# -----------------------------------------------------------------------------------------------------------#
def learn_task(dataSet, weightsLoadFile='', weightsSaveFile='', dropout_flag=False, n_steps=prm.default_n_steps):

    print('Number training of samples: ',  dataSet.train.num_examples)
    # -----------------------------------------------------------------------------------------------------------#
    #  Define Deterministic Learning Graph
    # -----------------------------------------------------------------------------------------------------------#

    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("graph"):
            # Create the model
            x = tf.placeholder(tf.float32, [None, input_size])

            # Define loss and optimizer
            yGT = tf.placeholder(tf.float32, [None, n_labels])  # Ground truth

            # Build the graph for the deep net
            net_out = func_standard_net.network_model(x, dropout_flag)
            net_out = tf.log(tf.clip_by_value(net_out, 1e-10, 1.0))  # avoid nan due to 0*log(0)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yGT, logits=net_out))

            # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables) # TODO: regularizer as hyperprior

            objective = loss

            # Variables lists:

            weightsVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph/net")

            # Learning rate:
            global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(prm.starter_learning_rate, global_step,
            #                                            prm.learnig_rate_decay_steps, prm.learning_rate_decay_rate,
            #                                            staircase=True)
            learning_rate = prm.learning_rate

            # Optimizer:
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(objective, var_list=weightsVars, global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(yGT, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver(var_list=weightsVars)

    tf.reset_default_graph()
    # -----------------------------------------------------------------------------------------------------------#
    #  Run Learning
    # -----------------------------------------------------------------------------------------------------------#

    with tf.Session(graph=graph) as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())

        # Get pre-trained weights (if available):
        if weightsLoadFile != '':
            print("Loading weights  from file: %s" % weightsLoadFile)
            saver.restore(sess, weightsLoadFile)

        for iStep in xrange(n_steps):

            # Draw random mini-batch:
            batch = dataSet.train.next_batch(prm.batch_size, shuffle=True)

            feed_dict = {x: batch[0], yGT: batch[1]}

            # Take gradient step:
            train_step.run(feed_dict=feed_dict)

            #  training performance evaluation:
            if iStep % 5000 == 0:
                train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                curr_objective = sess.run(objective, feed_dict=feed_dict)
                print('step %d, training batch accuracy %g, batch objective: %g' % (iStep, train_accuracy, curr_objective))

        # Evaluate on test set
        feed_dict = {x: dataSet.test.images, yGT: dataSet.test.labels}
        test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print('test accuracy %g' % test_accuracy)

        # Save weights variables:
        if weightsSaveFile != '':
            save_path = saver.save(sess, weightsSaveFile)
            print("Posterior saved in file: %s" % save_path)

    return test_accuracy

