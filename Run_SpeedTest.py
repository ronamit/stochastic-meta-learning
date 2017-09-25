

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
setting_name = 'Speed_Test'
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
#
n_batches_per_epoch = 469 # For MNIST 60,000 samples, batch size = 128
n_epoch = 200
n_steps_bayes = 469 * 200
#,
# ###################
#
# # Network params (currently only fully-connected)
# # List of the number of units (width) in each hidden layer:
# width_per_layer = [800, 800] # [800, 800]
#
# loss_type = 'L2_SVM' # 'softmax' \ 'hinge' \ 'L2_SVM'
#
# #  epsilons should be sampled ~ N(0,1):
# epsilonStdDefault = 1  # For debug set epsilons with 0.0 -> recovers standard NN
#
# # Learning Parameters:
# learning_rate = 1e-4  # 1e-4
#
# # Ratio of of steps for first stage (learning the posterior mean only):
# steps_stage_1_ratio = 0.05  # 0.1
#
# # Ratio of of steps out of the second stage with epsilon = 1:
# steps_with_full_eps_ratio = 0.2  # 0.5
#
# batch_size = 128
# # Note we need large enough batch to reduce the variance of the gradient estimate
# # since we are using the "Local Reparameterization Trick" the variance is proportional to 1/batch_size
#
# # Variables init
# random_init_std = 0.1
# bias_init = 0




#  Bayes Learning
# -----------------------------------------------------------------------------------------------------------#


print('---- Bayes-no-prior  learning for a single task...')
startRuntime = timeit.default_timer()
test_accuracy = learn_bayes_single_task.learn_task(data1, objective_type='Bayes_No_Prior', n_steps=n_steps_bayes)
stopRuntime = timeit.default_timer()
cmn.write_result('Bayes-no-prior  learning - Test Error: {0} %, Runtime: {1} [sec]'.
             format(100*(1-test_accuracy), stopRuntime - startRuntime),setting_name)
