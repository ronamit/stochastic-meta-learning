

from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

#-----------------------------------------------------------------------------------------------------------#
# Random Seed
#-----------------------------------------------------------------------------------------------------------#

# random_seed = 1235

#-----------------------------------------------------------------------------------------------------------#
# Definitions
#-----------------------------------------------------------------------------------------------------------#
FLAGS = None

# Data constants: (For MNIST)
input_size = 28 * 28
n_labels = 10

# Network params (currently only fully-connected)
# List of the number of units (width) in each hidden layer:
width_per_layer = [800, 800] # [800, 800]

loss_type = 'L2_SVM' # 'softmax' \ 'hinge' \ 'L2_SVM'

#  epsilons should be sampled ~ N(0,1):
epsilonStdDefault = 1  # For debug set epsilons with 0.0 -> recovers standard NN
# epsilonStdDefault = 1e-2

# nMC = 1# How many epsilons to draw to estimate the expectation by averaging (Monte-Carlo)

# Prior parameters
sigma_prior_init = 5 #  1  Initial sigma prior (if we don't optimize the prior, then it is fixed)

# Learning Parameters:
learning_rate = 1e-4  # 1e-4

# Total Number of steps:
default_n_steps = int(2e6) # 2000000

# Ratio of of steps for first stage (learning the posterior mean only):
steps_stage_1_ratio = 0.05  # 0.1

# Ratio of of steps out of the second stage with epsilon = 1:
steps_with_full_eps_ratio = 0.2  # 0.5

batch_size = 128
# Note we need large enough batch to reduce the variance of the gradient estimate
# since we are using the "Local Reparameterization Trick" the variance is proportional to 1/batch_size


# Variables init
random_init_std = 0.1
bias_init = 0

