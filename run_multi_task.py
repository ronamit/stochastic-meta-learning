

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

# -----------------------------------------------------------------------------------------------------------#
# Create training tasks:
# -----------------------------------------------------------------------------------------------------------#
n_tasks = 2
train_tasks_data = []
for _ in xrange(n_tasks):
    train_tasks_data.append(cmn.permute_pixels(orig_data))

# -----------------------------------------------------------------------------------------------------------#
#  Meta-training
# -----------------------------------------------------------------------------------------------------------#
prior_file_path = '/tmp/prior.ckpt'  # TODO: Date-time name

print('---- Meta-training ...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_multitask.learn_tasks(train_tasks_data,
                                                  objective_type='PAC_Bayes_McAllaster', prior_file_path=prior_file_path, mode='Meta_Training')
stopRuntime = timeit.default_timer()
cmn.write_result('Meta-training - Test Error: {0} %, Runtime: {1} [sec]'.format
                 (100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)

# -----------------------------------------------------------------------------------------------------------#
# Create test tasks:
# -----------------------------------------------------------------------------------------------------------#
n_tasks = 2
test_tasks_data = []
n_train_samples_in_test_tasks = 10000
for _ in xrange(n_tasks):
    test_tasks_data.append(cmn.permute_pixels(orig_data, n_train_samples_in_test_tasks))

print('---- Meta-Test set: {0} tasks of {1} samples'.format(n_tasks, n_train_samples_in_test_tasks))
# -----------------------------------------------------------------------------------------------------------#
#  Meta-testing
# -----------------------------------------------------------------------------------------------------------#
print('---- stadnard  single-task learning  in each test-task...')
test_accuracy_standard = 0
stopRuntime = timeit.default_timer()
for i_task in xrange(n_tasks):
    test_accuracy_standard += (1/n_tasks) *  model_standard_single_task.learn_task(test_tasks_data[i_task], dropoutFlag=False, n_steps=10000)

cmn.write_result('Meta-testing - Average Test Error of standard learning : {0} %, Runtime: {1} [sec]'.
                 format(100 * (1 - test_accuracy_standard), stopRuntime - startRuntime), setting_name)


print('---- Meta-testing ...')
# TODO: Print how many tasks and how many training examples
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_multitask.learn_tasks(train_tasks_data,
                                                  objective_type='PAC_Bayes_McAllaster', prior_file_path=prior_file_path, mode='Meta_Testing', n_steps=10000)
stopRuntime = timeit.default_timer()
cmn.write_result('Meta-testing - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)

# TODO: Run single task learning and calculate average error (implement early stop)


