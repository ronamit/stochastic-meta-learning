
from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import parameters as prm
import common as cmn
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
    n_samples_list = [x.train.num_examples for x in tasks_data]

    # -----------------------------------------------------------------------------------------------------------#
    #  Define the model's graph
    # -----------------------------------------------------------------------------------------------------------#

    graph = tf.Graph()

    with graph.as_default():
        with tf.variable_scope("graph"):
            # Note: We initialize the graph for all tasks using the same place-holder input of a single task
            # When the session is run, you need to fetch only the output relevant to the task in the input

            # The input (batch of samples for some task)
            x = tf.placeholder(tf.float32, [None, input_size])

            #  Ground truth labels:
            labels = tf.placeholder(tf.float32, [None, n_labels])

            # STD of epsilon for re-parametrization trick:
            eps_std = tf.placeholder(tf.float32, [])

            # Initialize the weight's prior variables:
            with tf.variable_scope('prior'):
                # call network with dummy inputs to add prior variables to graph
                sf.network_model('prior', init_source='constants', input=x, eps_std=1)

            # Build the graph for all tasks:

            total_complexity = 0
            intra_task_objectives = []
            intra_task_accuaracy = []

            for i_task in xrange(n_tasks):
                task_tag = 'posterior_' + str(i_task)

                with tf.variable_scope(task_tag):

                    # Run net of current task:
                    net_out = sf.network_model(task_tag, init_source='random', input=x, eps_std=eps_std)
                    # Note: x is a placeholder that should be fed the inputs of the task which outputs are fetched

                    # The empirical loss:
                    average_loss = tf.reduce_mean(cmn.loss_function(labels, net_out))

                    # The Kullback-Leibler divergence between posterior and prior:
                    kl_dist = sf.calculate_kl_dist(task_tag, 'prior')

                    n_samples = n_samples_list[i_task]

                    # Complexity term:
                    complex_term = sf.single_task_complexity(objective_type, n_samples, kl_dist)
                    total_complexity += complex_term

                    # The intra-task objective:
                    task_obj = average_loss + complex_term
                    intra_task_objectives.append(task_obj)

                    # Compare net output to true labels (for evaluation):
                    correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    intra_task_accuaracy.append(accuracy)


            # Add the hyper-prior term:
            hyper_prior_factor = 0.0001 * (1 / np.sqrt(n_tasks))
            regularizer = tf.contrib.layers.l2_regularizer(scale=hyper_prior_factor)
            hyper_prior_term = hyper_prior_factor * tf.contrib.layers.apply_regularization(regularizer, tf.get_collection('prior'))
            # hyper_prior_term = hyper_prior_factor * sf.calc_param_norm('prior')
            total_complexity += hyper_prior_term

            # Learning rate:
            learning_rate = prm.learning_rate

            # Optimization step for the prior:
            prior_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='graph/prior')
            prior_step = tf.train.AdamOptimizer(learning_rate).minimize(total_complexity, var_list=prior_vars_list)

            # Optimization step for each task's posterior:
            task_posterior_steps = []
            for i_task in xrange(n_tasks):
                posterior_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'graph/posterior_'+str(i_task))
                task_posterior_steps.append(tf.train.AdamOptimizer(learning_rate).
                                            minimize(intra_task_objectives[i_task], var_list=posterior_vars_list))
    # end graph

    tf.reset_default_graph()

    # Add ops to save and restore the prior variables:
    prior_saver = tf.train.Saver(prior_vars_list)

    # -----------------------------------------------------------------------------------------------------------#
    #  Run Learning
    # -----------------------------------------------------------------------------------------------------------#

    train_eval_interval = 5000 # Print training performance evaluation after each interval of steps

    n_steps_stage_1 = int(n_steps * prm.steps_stage_1_ratio)
    n_steps_stage_2 = n_steps - n_steps_stage_1
    n_steps_with_full_eps_std = int(n_steps_stage_2 * prm.steps_with_full_eps_ratio)

    prior_update_interval = 5

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

            avg_accuracy, avg_intra_obj = 0, 0
            # Take posterior step in each task:
            for i_task in xrange(n_tasks):
                # Collect random mini-batch from current task:
                batch = tasks_data[i_task].train.next_batch(prm.batch_size, shuffle=True)
                feed_dict = {x: batch[0], labels:batch[1], eps_std: cur_eps_std}
                # Take gradient step:
                task_posterior_steps[i_task].run(feed_dict=feed_dict)

                # training performance evaluation:
                if i_step % train_eval_interval == 0:
                    curr_obj, curr_accuracy = sess.run([intra_task_objectives[i_task],
                                                         intra_task_accuaracy[i_task]], feed_dict=feed_dict)
                    avg_accuracy += (1/n_tasks) * curr_accuracy
                    avg_intra_obj += (1 / n_tasks) * curr_obj

            # Take prior step:
            if mode == 'Meta_Training' and (i_step % prior_update_interval == 0):
                prior_step.run() # no need for data feed

            #  training performance evaluation:
            if i_step % train_eval_interval == 0:
                curr_total_complexity, curr_hyper_prior_term = sess.run([total_complexity, hyper_prior_term])
                avg_intra_complex = (curr_total_complexity - curr_hyper_prior_term)/n_tasks
                avg_empiric_loss = avg_intra_obj - avg_intra_complex
                print('step %d, eps: %g, avg-accuracy %g, avg-empiric-loss: %g, avg-intra-complexity: %g, hyper-prior: %g' %
                      (i_step, cur_eps_std, avg_accuracy, avg_empiric_loss,
                      avg_intra_complex, curr_hyper_prior_term))
        # end optimization steps

        # Evaluate on the test-sets of training-tasks:
        # (Note: the test sets are not available for the meta-learner)
        test_accuracy = 0
        for i_task in xrange(n_tasks):
            test_input = tasks_data[i_task].test.images
            test_label = tasks_data[i_task].test.labels
            # Evaluate on test with epsilon = 0 (maximum of posterior)
            feed_dict = {x: test_input, labels: test_label, eps_std: 0}
            test_accuracy += (1 / n_tasks) * sess.run(intra_task_accuaracy[i_task], feed_dict=feed_dict)
        print('test accuracy %g' % test_accuracy)

        # Save prior variables:
        if mode == 'Meta_Training' and not prior_file_path == '':
            save_path = prior_saver.save(sess, prior_file_path)
            print("Prior saved in file: %s" % save_path)

    return test_accuracy

