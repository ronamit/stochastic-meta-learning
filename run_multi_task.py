

from __future__ import absolute_import, division, print_function

import timeit

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.examples.tutorials.mnist import input_data

import common as cmn
import model_bayes_multitask
import model_standard_single_task
# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#
setting_name = 'Multi_Task'
run_name = ''
cmn.save_to_archive(setting_name, run_name)

# TODO: option to set fixed random streams..

# -----------------------------------------------------------------------------------------------------------#
# Import data:
# -----------------------------------------------------------------------------------------------------------#
data_dir = '/tmp/tensorflow/mnist/input_data'
orig_data = input_data.read_data_sets(data_dir, one_hot=True)   # Load MNIST
#  Append the validation to training data:
orig_data.train._labels = np.append(orig_data.train.labels, orig_data.validation.labels, axis=0)
orig_data.train._images = np.append(orig_data.train.images, orig_data.validation.images, axis=0)
orig_data.train._num_examples = orig_data.train.images.shape[0]

with tf.Session() as sess:
    pass

n_steps_bayes = int(2e6)
n_steps_standard = int(1e5)

# n_steps_bayes = int(2e3)
# n_steps_standard = int(1e2)

# -----------------------------------------------------------------------------------------------------------#
# Create training tasks:
# -----------------------------------------------------------------------------------------------------------#
n_tasks_train = 2
train_tasks_data = []
for _ in xrange(n_tasks_train):
    train_tasks_data.append(cmn.permute_pixels(orig_data))
cmn.write_result('---- Meta-Training set: {0} tasks of {1} training samples'.format(n_tasks_train, orig_data.train.num_examples), setting_name)
# # -----------------------------------------------------------------------------------------------------------#
# #  Meta-training
# # -----------------------------------------------------------------------------------------------------------#
prior_file_path = '/tmp/prior.ckpt'  # TODO: Date-time name
#
print('---- Meta-training ...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_multitask.learn_tasks(train_tasks_data,
                                                  objective_type='PAC_Bayes_McAllaster', prior_file_path=prior_file_path,
                                                  mode='Meta_Training', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Meta-training - Test Error: {0} %, Runtime: {1} [sec]'.format
                 (100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)

# -----------------------------------------------------------------------------------------------------------#
# Create test tasks:
# -----------------------------------------------------------------------------------------------------------#
n_tasks_test = 1
test_tasks_data = []
n_train_samples_in_test_tasks = 10000
for _ in xrange(n_tasks_test):
    test_tasks_data.append(cmn.permute_pixels(orig_data, train_samples_lim=n_train_samples_in_test_tasks))

cmn.write_result('---- Meta-Test set: {0} tasks of {1} training samples'.format(n_tasks_test, n_train_samples_in_test_tasks), setting_name)
# -----------------------------------------------------------------------------------------------------------#
#  Meta-testing
# -----------------------------------------------------------------------------------------------------------#
print('---- Standard Meta-testing ...')
print('---- separate standard  single-task learning  in each test-task...')
test_accuracy_standard_avg = 0

startRuntime = timeit.default_timer()
for i_task in xrange(n_tasks_test):
    test_accuracy_standard = model_standard_single_task.learn_task(
        test_tasks_data[i_task], dropout_flag=False, n_steps=n_steps_standard)
    cmn.write_result('Meta-testing - Test Error in task {0} - standard learning : {1} %'.
                     format(i_task, 100 * (1 - test_accuracy_standard)), setting_name)
    test_accuracy_standard_avg += (1/n_tasks_test) * test_accuracy_standard
stopRuntime = timeit.default_timer()

cmn.write_result('Meta-testing - standard learning - Average Test Error : {0} %, Runtime: {1} [sec]'.
                 format(100 * (1 - test_accuracy_standard_avg), stopRuntime - startRuntime), setting_name)

print('---- Bayesian Meta-testing ...')
# TODO: Print how many tasks and how many training examples
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_multitask.learn_tasks(
    test_tasks_data,objective_type='PAC_Bayes_McAllaster',
    prior_file_path=prior_file_path, mode='Meta_Testing', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Meta-testing - Bayesian - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)

# TODO: Run single task learning and calculate average error (implement early stop)


