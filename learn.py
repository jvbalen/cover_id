
from __future__ import division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf


# simple 2D convolutional layer
def conv_layer(x, shape=(4,1),
               n_filters=None,
               strides=[1, 1, 1, 1],
               padding='SAME',
               sigma=tf.nn.relu):
    n_channels = int(x.get_shape()[-1])
    if n_filters is None:
        n_filters = n_channels * 2

    W = weight_variable([shape[0], shape[1], n_channels, n_filters])
    b = bias_variable([n_filters])
    
    h = b + tf.nn.conv2d(x, W, strides=strides, padding=padding)
    
    return sigma(h)


# simple 2D max pool layer
def max_pool_layer(x, shape=(4, 1), 
                   strides=None, 
                   padding='SAME'):
    
    ksize = [1, shape[0], shape[1], 1]
    if strides is None:
        strides = ksize
        
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)
        

# simple fully-connected layer
def fully_connected_layer(x, n_nodes,
                          sigma=tf.nn.tanh):
    
    batch_size, dim1, dim2, n_channels = x_A.get_shape()
    n_nodes_in = int(dim1) * int(dim2) * int(n_channels)
    x_flat = tf.reshape(x, [-1, n_nodes_in])

    W = weight_variable([n_nodes_in, n_nodes])
    b = bias_variable([n_nodes])

    return sigma(tf.matmul(x_flat, W) + b)


def build_butterfly_network(x_A, x_B, list_of_layers):
    network_A = [x_A]
    network_B = [x_B]
    for layer in list_of_layers:
        layer_type, params = layer
        network_A.append(layer_type(network_A[-1], **params))
        network_B.append(layer_type(network_B[-1], **params))
    return network_A, network_B


def weight_variable(shape, weight_scale=0.1):
    initial = tf.truncated_normal(shape, stddev=weight_scale)
    return tf.Variable(initial)


def bias_variable(shape, weight_scale=0.1):
    initial = tf.constant(weight_scale, shape=shape)
    return tf.Variable(initial)


def get_batches(arrays, batch_size=50):
    """Batch generator, no shuffling.
    
    Args:
        arrays (list): list of arrays. Arrays should have equal length
        batch_size (int): number of examples per batch
        
    Yields:
        list: list of song pairs of length batch_size
        
    Usage:
    >>> batches = get_batches([X, Y], batch_size=50)
    >>> x, y = batches.next()
    """
    array_lengths = [len(array) for array in arrays]
    n_examples = array_lengths[0]
    if not np.all(np.array(array_lengths) == n_examples):
        raise ValueError('Arrays must have the same length.')
    start = 0
    while True:
        start = np.mod(start, n_examples)
        stop = start + batch_size
        batch = [np.take(array, range(start, stop), axis=0, mode='wrap') for array in arrays]
        start = stop
        yield batch


def approx_bhattacharyya(squared_dists, is_cover):
    """Approximate bhattacharyya distance between cover and non-cover distances.
    
    Similar to Mahalanobis distance, but for distributions with different variances.
    Assumes normality, hence approximate (distances are bound by 0).
    """
    pair_dists = np.sqrt(squared_dists[np.where(is_cover==1)])
    non_pair_dists = np.sqrt(squared_dists[np.where(is_cover==0)])
    
    mu_pairs, sigma2_pairs = np.mean(pair_dists), np.var(pair_dists)
    mu_non_pairs, sigma2_non_pairs = np.mean(non_pair_dists), np.var(non_pair_dists)

    bhatt = (0.25 * np.log(0.25 * (sigma2_pairs/sigma2_non_pairs + sigma2_non_pairs/sigma2_pairs + 2)) +
             0.25 * (mu_pairs - mu_non_pairs)**2 / (sigma2_pairs + sigma2_non_pairs))
    return bhatt, mu_pairs, mu_non_pairs


def report(train_log, step, train_batch, test_batch, metrics):
    
    train_metrics = ['train_' + metric.__name__ for metric in metrics]
    test_metrics = ['test_' + metric.__name__ for metric in metrics]

    # initialization short-hand
    if type(train_log) is int:
        n_epoques = train_log
        train_log = pd.DataFrame(data = np.zeros((n_epoques, len(metrics))),
                                 columns=train_metrics + test_metrics)
    
    # train and test feeds
    train_feed = {x_A:train_batch[0], x_B:train_batch[1], y_: train_batch[2]}
    test_feed = {x_A:test_batch[0], x_B:test_batch[1], y_: test_batch[2]}
    
    # compute and log metrics
    train_log.loc[step, train_metrics] = sess.run(metrics, feed_dict=train_feed)
    train_log.loc[step, test_metrics] = sess.run(metrics, feed_dict=test_feed)

    # print metrics
    print(train_log[step:])
    
    return train_log