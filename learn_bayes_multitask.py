
from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import parameters as prm
import func_bayes_net as sf
from common import subset_with_substring
#-----------------------------------------------------------------------------------------------------------#
#   global parameters
#-----------------------------------------------------------------------------------------------------------#
input_size = prm.input_size
n_labels = prm.n_labels

# -----------------------------------------------------------------------------------------------------------#
#  Multi-Task Bayesian Learning Function
# -----------------------------------------------------------------------------------------------------------#


def learn_tasks(tasks_data, objective_type, prior_file_path='', mode='', n_steps=prm.default_n_steps):


    if mode not in ['Meta_Training', 'Meta_Testing']:
        raise ValueError('Invalid mode')

    n_tasks = len(tasks_data)
    print('Number training of tasks: ', n_tasks)

    # list of the number of training examples in each task:
    n_samples_list =  [x.train.num_examples for x in tasks_data]

    # -----------------------------------------------------------------------------------------------------------#
    #  Define the model's graph
    # -----------------------------------------------------------------------------------------------------------#

    graph = tf.Graph()

    with graph.as_default():
        with tf.variable_scope("graph"):

            # The input (batches of samples for each task)
            x = tf.placeholder(tf.float32, [n_tasks, None, input_size])

            #  Ground truth labels:
            labels = tf.placeholder(tf.float32, [n_tasks, None, n_labels])

            # STD of epsilon for re-parametrization trick:
            eps_std = tf.placeholder(tf.float32, [])

            # Initialize the weight's prior variables:
            with tf.variable_scope('prior'):
                # call network with dummy inputs to add prior variables to graph
                sf.network_model('prior', init_source='constants', input=x[0], eps_std=1)

            # Build the total objective from all tasks:
            objective = 0
            total_kl, multitask_avg_loss = 0, 0
            avg_accuracy = 0

            for i_task in xrange(n_tasks):
                task_tag = 'posterior_' + str(i_task)
                # For each task, apply the net with  appropriate inputs and weights posterior:
                with tf.variable_scope(task_tag):

                    net_out = sf.network_model(task_tag, init_source='random', input=x[i_task], eps_std=eps_std)
                    net_out = tf.log(tf.clip_by_value(net_out, 1e-10, 1.0))  # avoid nan due to 0*log(0)

                    # The empirical loss:
                    average_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=labels[i_task], logits=net_out))

                    # The Kullback-Leibler divergence between posterior and prior:
                    kl_dist = sf.calculate_kl_dist(task_tag, 'prior')

                    n_samples = n_samples_list[i_task]

                    # Add the contribution of current to the total objective:
                    with tf.control_dependencies([tf.assert_non_negative(average_loss)]):  # debug assertion
                        objective += sf.single_task_objective(objective_type, average_loss, n_samples, kl_dist)

                    # Compare net output to true labels:
                    correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(labels[i_task], 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    avg_accuracy += (1/n_tasks) * accuracy

                    # for debug:
                    total_kl += kl_dist
                    multitask_avg_loss += (1/n_tasks) * average_loss

            # Add the hyper-prior term:
            hyper_prior_factor = 0.01 * (1 / np.sqrt(n_tasks))
            objective += hyper_prior_factor * sf.calc_param_norm('prior')

            # regularizer = tf.contrib.layers.l2_regularizer(scale=hyper_prior_factor)
            # reg_variables = tf.get_collection('prior')
            # reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            # objective += hyper_prior_factor * reg_term

            # Learning rate:
            learning_rate = prm.learning_rate

            # Trainable variables lists:
            all_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            prior_vars_list = subset_with_substring(all_vars_list, 'graph/prior')
            posterior_vars_list = subset_with_substring(all_vars_list, 'graph/posterior_')

            # Optimization step:
            posterior_step = tf.train.AdamOptimizer(learning_rate).minimize(objective, var_list=posterior_vars_list)
            prior_step = tf.train.AdamOptimizer(learning_rate).minimize(objective, var_list=prior_vars_list)

    tf.reset_default_graph()

    # Add ops to save and restore the prior variables:
    prior_saver = tf.train.Saver(prior_vars_list)

    # -----------------------------------------------------------------------------------------------------------#
    #  Run Learning
    # -----------------------------------------------------------------------------------------------------------#


    n_steps_stage_1 = int(n_steps * prm.steps_stage_1_ratio)
    n_steps_stage_2 = n_steps - n_steps_stage_1
    n_steps_with_full_eps_std = int(n_steps_stage_2 * prm.steps_with_full_eps_ratio)


    with tf.Session(graph=graph) as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())


        # In meta-testing, we used the pre-learned prior to learn new tasks:
        if mode == 'Meta_Testing':
            prior_saver.restore(sess, prior_file_path)
            print("Prior restored from: {0}".format(prior_file_path))


        for i_step in xrange(n_steps):

            # We gradually increase epsilon's STD from 0 to 1.
            # The reason is that using 1 from the start results in high variance gradients.
            if i_step >= n_steps_stage_1:
                cur_eps_std = prm.epsilonStdDefault * (i_step - n_steps_stage_1) / (n_steps_stage_2 - n_steps_with_full_eps_std)
            else:
                cur_eps_std = 0.0
            cur_eps_std = min(max(cur_eps_std, 0.0), 1)  # keep in [0,1]

            # Collect random mini-batch from all tasks:
            X = np.zeros([n_tasks, prm.batch_size, input_size])
            Y = np.zeros([n_tasks, prm.batch_size, n_labels])
            for i_task in xrange(n_tasks):
                batch = tasks_data[i_task].train.next_batch(prm.batch_size, shuffle=True)
                X[i_task] = batch[0]
                Y[i_task] = batch[1]

            feed_dict = {x: X, labels:Y, eps_std: cur_eps_std}

            # Take gradient step:
            posterior_step.run(feed_dict=feed_dict)
            if mode == 'Meta_Training':
                prior_step.run(feed_dict=feed_dict)

            #  training performance evaluation:
            if i_step % 5000 == 0:
                (train_accuracy, curr_objective, cur_total_kl, cur_multitask_avg_loss)= \
                    sess.run([avg_accuracy, objective, total_kl, multitask_avg_loss], feed_dict=feed_dict)

                print('step %d, eps: %g, avg accuracy %g, objective: %g, ' %
                      (i_step, cur_eps_std, train_accuracy, curr_objective) +
                      'total kl: %g, avg loss: %g' % (cur_total_kl, cur_multitask_avg_loss))

        # Evaluate on test set:
        # assuming same number of test samples in all tasks
        X = np.zeros((n_tasks, tasks_data[0].test.num_examples, input_size))
        Y = np.zeros((n_tasks, tasks_data[0].test.num_examples, n_labels))
        for i_task in xrange(n_tasks):
            X[i_task, :, :] = tasks_data[i_task].test.images
            Y[i_task, :, :] = tasks_data[i_task].test.labels
        # Evaluate on test with epsilon = 0 (maximum of posterior)
        feed_dict = {x: X, labels: Y, eps_std: 0}
        test_accuracy = sess.run(avg_accuracy, feed_dict=feed_dict)
        print('test accuracy %g' % test_accuracy)

        # Save prior variables:
        if mode == 'Meta_Training' and not prior_file_path == '':
            save_path = prior_saver.save(sess, prior_file_path)
            print("Prior saved in file: %s" % save_path)

    return test_accuracy

