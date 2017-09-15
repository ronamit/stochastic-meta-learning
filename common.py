



from __future__ import absolute_import, division, print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tensorflow as tf
import argparse
import sys
import itertools
import glob
import os
import shutil
from copy import deepcopy
import parameters as prm

# -----------------------------------------------------------------------------------------------------------#
#  General TensorFlow functions
# -----------------------------------------------------------------------------------------------------------#

def get_reused_variable(scope, *args, **kwargs):
    ''' If the variables has not been defined yet - create it, otherwise - reuse it'''
    try:
        var = tf.get_variable(*args, **kwargs)
    except ValueError:
        scope.reuse_variables()
        var = tf.get_variable(*args, **kwargs)
    return var


def add_tensors_to_collection(collection, tensor_list):
    for tensor in tensor_list:
        tf.add_to_collection(collection, tensor)


def subset_with_substring(tensor_list, substring):
    return [a for a in tensor_list if substring in a.name]

def get_var_from_list(tensor_list, var_name):
    return [var for var in tensor_list if var.name == var_name][0]


def loss_function(labels, net_out):
    '''' compute loss for a batch of data samples'''

    if prm.loss_type == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=net_out)

    elif prm.loss_type == 'hinge':
        return tf.losses.hinge_loss(labels=labels, logits=net_out)

    elif prm.loss_type == 'L2_SVM':
        return tf.square(tf.losses.hinge_loss(labels=labels, logits=net_out))


    else:
        raise ValueError('Invalid loss_type')




# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#

def write_result(str, setting_name):

    print(str)
    with open(setting_name + '.out', 'a') as f:
        print(str, file=f)

        def save_to_archive(setting_name, run_name=''):
            start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            dir_name = setting_name + '_' + start_time_str
            # Create backup of code
            source_dir = os.getcwd()
            dest_dir = source_dir + '/Results_Archive/' + dir_name
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for filename in glob.glob(os.path.join(source_dir, '*.*')):
                shutil.copy(filename, dest_dir)

            write_result('-' * 30 + start_time_str + '-' * 30 + run_name, setting_name)


def save_to_archive(setting_name, run_name=''):
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dir_name = setting_name + '_' + start_time_str
    # Create backup of code
    source_dir = os.getcwd()
    dest_dir = source_dir + '/Results_Archive/' + dir_name
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename in glob.glob(os.path.join(source_dir, '*.*')):
        shutil.copy(filename, dest_dir)

    write_result('-' * 30 + start_time_str + '-' * 30 + run_name, setting_name)


# -----------------------------------------------------------------------------------------------------------#
#  Data Handling Functions
# -----------------------------------------------------------------------------------------------------------#

def mnist_imshow(img):
    '''    mnist imshow convenience function
    input is a 1D array of length 784
    Reference: https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb '''
    plt.imshow(img.reshape([28, 28]), cmap="gray")
    plt.axis('off')


def limit_data(data, samples_lim):
    n_samples_orig = data.num_examples
    n_samples_new = min(n_samples_orig, samples_lim)
    samples_permute = np.random.choice(n_samples_orig, n_samples_new, replace=False)
    data._images = data.images[samples_permute, :]
    data._labels = data.labels[samples_permute, :]
    data._num_examples = n_samples_new


def permute_pixels(orig_data, train_samples_lim=-1):
    '''     return a new mnist dataset w/ pixels randomly permuted
    Reference: https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb '''
    n_pixels = orig_data.train.images.shape[1]
    pixel_permute = np.random.permutation(n_pixels)
    new_data = deepcopy(orig_data)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(new_data, set_name) # shallow copy
        this_set._images = this_set.images[:, pixel_permute]

    if train_samples_lim != -1:
        limit_data(new_data.train, train_samples_lim)

    return new_data


def permute_labels(orig_data, train_samples_lim=-1):
    '''     return a new mnist dataset w/ labels randomly permuted
    Reference: https://github.com/ariseff/overcoming-catastrophic/blob/master/experiment.ipynb '''
    n_labels = orig_data.train.labels.shape[1]
    labels_permute = np.random.permutation(n_labels)
    new_data = deepcopy(orig_data)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(new_data, set_name) # shallow copy
        this_set._labels = this_set.images[:, labels_permute]

    if train_samples_lim != -1:
        limit_data(new_data.train, train_samples_lim)

    return new_data


