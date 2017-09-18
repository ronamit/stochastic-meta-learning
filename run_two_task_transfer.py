
from __future__ import absolute_import, division, print_function

import timeit

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.examples.tutorials.mnist import input_data

import common as cmn
import learn_bayes_multitask
import learn_standard_single_task
# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#
setting_name = 'Two_Task_Trasnfer'
run_name = ''
cmn.save_to_archive(setting_name, run_name)

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

n_steps_bayes = int(2e5)
n_steps_standard = int(1e4)

weights_file = '/tmp/weights11.ckpt'
prior_file_path = '/tmp/prior11.ckpt'
#  -----------------------------------------------------------------------------------------------------------#
# Create source + target tasks:
# -----------------------------------------------------------------------------------------------------------#
source_task = orig_data
train_samples_lim=1000
target_task = cmn.permute_pixels(orig_data, train_samples_lim=train_samples_lim)

# -----------------------------------------------------------------------------------------------------------#
# # standard learning
# # -----------------------------------------------------------------------------------------------------------#

print('---- Standard learning on source task...')

startRuntime = timeit.default_timer()
test_accuracy_standard = learn_standard_single_task.learn_task(
    dataSet=source_task, weights_save_file=weights_file, dropout_flag=False, n_steps=n_steps_standard)
stopRuntime = timeit.default_timer()
cmn.write_result('Source task - standard learning  - Average Test Error : {0} %, Runtime: {1} [sec]'.
                 format(100 * (1 - test_accuracy_standard), stopRuntime - startRuntime), setting_name)

print('---- Standard learning on target task using transfered initial point...')
startRuntime = timeit.default_timer()
test_accuracy_standard = learn_standard_single_task.learn_task(
    dataSet=target_task, weights_restore_file=weights_file,dropout_flag=False, n_steps=n_steps_standard)
stopRuntime = timeit.default_timer()
cmn.write_result('Target task- standard learning - with transfer -  Average Test Error : {0} %, Runtime: {1} [sec]'.
                 format(100 * (1 - test_accuracy_standard), stopRuntime - startRuntime), setting_name)


print('---- Standard learning on target task -no transfer...')
startRuntime = timeit.default_timer()
test_accuracy_standard = learn_standard_single_task.learn_task(
    dataSet=target_task, weights_restore_file='',dropout_flag=False, n_steps=n_steps_standard)
stopRuntime = timeit.default_timer()
cmn.write_result('Target task- standard learning -no transfer - Average Test Error : {0} %, Runtime: {1} [sec]'.
                 format(100 * (1 - test_accuracy_standard), stopRuntime - startRuntime), setting_name)

# -----------------------------------------------------------------------------------------------------------#
# Bayesian learning
# -----------------------------------------------------------------------------------------------------------#

traing_tasks_data = [source_task]
print('---- Bayesian learning on source task...')
startRuntime = timeit.default_timer()
test_accuracy = learn_bayes_multitask.learn_tasks(
    traing_tasks_data,objective_type='Variational_Bayes',
    prior_file_path=prior_file_path, mode='Meta_Training', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Meta-Training - Bayesian - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)


test_tasks_data = [target_task]
print('---- Bayesian learning on Target task with transferred prior..')
startRuntime = timeit.default_timer()
test_accuracy = learn_bayes_multitask.learn_tasks(
    test_tasks_data, objective_type='Variational_Bayes',
    prior_file_path=prior_file_path, mode='Meta_Testing', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Meta-Testing - Bayesian - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)
