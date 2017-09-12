

from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import timeit
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import parameters as prm
import common as cmn
import model_standard_single_task
import model_bayes_single_task

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#
setting_name = 'Single_Task'
run_name = ''
cmn.save_to_archive(setting_name, run_name)


# TODO: option to set fixed random streams..

# -----------------------------------------------------------------------------------------------------------#
# Import data:
# -----------------------------------------------------------------------------------------------------------#
data_dir = '/tmp/tensorflow/mnist/input_data'
data1 = input_data.read_data_sets(data_dir, one_hot=True)   # Load MNIST
#  Append the validation to training data:
data1.train._labels = np.append(data1.train.labels, data1.validation.labels, axis=0)
data1.train._images = np.append(data1.train.images, data1.validation.images, axis=0)
data1.train._num_examples = data1.train.images.shape[0]

with tf.Session() as sess:
    pass

# --------------------------------------------------------------------------------------#
# Run experiments
# --------------------------------------------------------------------------------------#

# Variational Bayes Learning
# -----------------------------------------------------------------------------------------------------------#
print('---- Variational Bayes learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_single_task.learn_task(data1, objective_type='Variational_Bayes')
stopRuntime = timeit.default_timer()
cmn.write_result('Variational Bayes learning - Test Error: {0} %, Runtime: {1} [sec]'
             .format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)

# -----------------------------------------------------------------------------------------------------------#
# PAC-Bayes Learning
# -----------------------------------------------------------------------------------------------------------#
print('---- PAC-Bayesian McAllaster learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_single_task.learn_task(data1, objective_type='PAC_Bayes_McAllaster')
stopRuntime = timeit.default_timer()
cmn.write_result('PAC-Bayesian McAllaster learning - Test Error: {0} %, Runtime: {1} [sec]'.
             format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)


print('---- PAC-Bayesian Pentina learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_single_task.learn_task(data1, objective_type='PAC_Bayes_Pentina')
stopRuntime = timeit.default_timer()
cmn.write_result('PAC-Bayesian Pentina learning - Test Error: {0} %, Runtime: {1} [sec]'.
             format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)


print('---- Bayes-no-prior  learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_single_task.learn_task(data1, objective_type='Bayes_No_Prior')
stopRuntime = timeit.default_timer()
cmn.write_result('Bayes-no-prior  learning - Test Error: {0} %, Runtime: {1} [sec]'.
             format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)


# Seeger
print('---- PAC-Bayesian Seeger learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_bayes_single_task.learn_task(data1, objective_type='PAC_Bayes_Seeger')
stopRuntime = timeit.default_timer()
cmn.write_result('PAC-Bayesian Seeger learning - Test Error: {0} %, Runtime: {1} [sec]'.
             format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)


# # --------   ---------------------------------------------------------------------------------------------------#
# # Standard Learning
# -----------------------------------------------------------------------------------------------------------#

print('---- Standard learning (non-Bayesian net) for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_standard_single_task.learn_task(data1, weightsLoadFile='', weightsSaveFile='')
stopRuntime = timeit.default_timer()
cmn.write_result('Standard net - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)


print('---- Standard (non-Bayesian net) + dropout learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = model_standard_single_task.learn_task(data1, weightsLoadFile='', weightsSaveFile='', dropout_flag=True)
stopRuntime = timeit.default_timer()
cmn.write_result('Standard net + dropout  - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)
