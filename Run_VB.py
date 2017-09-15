

from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import timeit
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import parameters as prm
import common as cmn
import learn_standard_single_task
import learn_bayes_single_task

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

n_steps_bayes = int(2e6)
n_steps_standard = int(1e5)



# # --------   ---------------------------------------------------------------------------------------------------#
# # Standard Learning
# -----------------------------------------------------------------------------------------------------------#

print('---- Standard learning (non-Bayesian net) for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = learn_standard_single_task.learn_task(data1, n_steps=n_steps_standard)
stopRuntime = timeit.default_timer()
cmn.write_result('Standard net - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)


print('---- Standard (non-Bayesian net) + dropout learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = learn_standard_single_task.learn_task(data1, dropout_flag=True, n_steps=n_steps_standard)
stopRuntime = timeit.default_timer()
cmn.write_result('Standard net + dropout  - Test Error: {0} %, Runtime: {1} [sec]'.
                 format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)



#  Bayes Learning
# -----------------------------------------------------------------------------------------------------------#
print('---- Variational Bayes learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = learn_bayes_single_task.learn_task(data1, objective_type='Variational_Bayes', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Variational Bayes learning - Test Error: {0} %, Runtime: {1} [sec]'
             .format(100*(1-test_accuracy), stopRuntime - startRuntime), setting_name)


print('---- Bayes-no-prior  learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = learn_bayes_single_task.learn_task(data1, objective_type='Bayes_No_Prior', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Bayes-no-prior  learning - Test Error: {0} %, Runtime: {1} [sec]'.
             format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)
