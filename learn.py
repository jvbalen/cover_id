
from __future__ import division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf


class siamese_network():

    def __init__(self, input_shape=(512,12)):
        """

        """
        n_frames, n_bins = input_shape

        self.x_A = tf.placeholder('float', shape=[None, n_frames, n_bins], name='x_A')
        self.x_B = tf.placeholder('float', shape=[None, n_frames, n_bins], name='x_B')

        self.subnet_A = [tf.reshape(self.x_A, [-1, n_frames, n_bins, 1])]
        self.subnet_B = [tf.reshape(self.x_B, [-1, n_frames, n_bins, 1])]

        self.is_cover = tf.placeholder('float', shape=[None], name='is_cover')

        self.train_log = None
        self.log_count = 0
    
    def add_conv_layer(self, shape=(4,1),
                             n_filters=None,
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             sigma=tf.nn.relu):
        """Add a simple 2D convolutional layer to each subnet."""
        x_A = self.subnet_A[-1]
        x_B = self.subnet_B[-1]
        assert np.all(x_A.get_shape() == x_A.get_shape())

        n_channels = int(x_A.get_shape()[-1])

        # default n_filters is 2 * previous n_filters
        if n_filters is None:
            n_filters = n_channels * 2

        W = self.weight_variable([shape[0], shape[1], n_channels, n_filters])
        b = self.bias_variable([n_filters])

        for subnet in [self.subnet_A, self.subnet_B]:
            x = subnet[-1]
            h = sigma(b + tf.nn.conv2d(x, W, strides=strides, padding=padding))
            subnet.append(h)

    # def add_attention(self):

    #     x_A = self.subnet_A[-1]
    #     shape_x = x_A.get_shape()
    #     batch_size, n_frames, n_bins, n_channels = [int(dim) for dim in shape_x]

    #     W = tf.weight_variable([1, n_bins, n_channels, 1])
    #     b = tf.bias_variable([1])

    #     for subnet in [self.subnet_A, self.subnet_B]:
    #         att = tf.softmax(tf.conv2d(W), W, strides=[1,1,1,1], padding='VALID')
    #         h = 
    #         subnet.append(h)


    def add_max_pool_layer(self, shape=(4, 1), 
                                 strides=None, 
                                 padding='SAME'):
        """Add a 2D max pool layer to each subnet."""
        for subnet in [self.subnet_A, self.subnet_B]:
            x = subnet[-1]
            ksize = [1, shape[0], shape[1], 1]
            if strides is None:
                strides = ksize
            h = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)
            subnet.append(h)

    def add_fully_connected_layer(self, n_nodes, sigma=tf.nn.tanh):
        """Add a fully-connected layer to each subnet.

        Args:
            n_nodes (int): number of nodes.
            sigma (tf.op): non-linearity as a Tensorflow function,
                e.g., tf.nn.tanh or tf.nn.relu.
        """
        x_A = self.subnet_A[-1] 
        x_B = self.subnet_B[-1]
        assert np.all(x_A.get_shape() == x_A.get_shape())

        n_nodes_in = np.prod([int(dim) for dim in x_A.get_shape()[1:]])

        W = self.weight_variable([n_nodes_in, n_nodes])
        b = self.bias_variable([n_nodes])

        for subnet in [self.subnet_A, self.subnet_B]:
            x = subnet[-1]
            x_flat = tf.reshape(x, [-1, n_nodes_in])

            h = sigma(tf.matmul(x_flat, W) + b)
            subnet.append(h)

    def add_matmul_layer(self,
                         filter_len=4,
                         n_filters=8,
                         strides=[1, 1, 1, 1],
                         sigma=tf.nn.tanh):
        """MATMUL layer.

        For each subnet:
        - makes 2 copies of the last layer
        - performs convolution with shape (filter_len, 1) on one of the copies
            using n_filters different filters
        - transpose sother copy
        - for each filter, matrix multiplies convolved copy with transposed copy
            (summing over the 'height' dimension shape[1])
        - flattens result and applies nonlinearilty sigma

        Requires that last layer has shape [batch_size, n_frames, 1, n_channels].
        Output has shape [batch_size, n_filters * n_channels**2].

        Args:
            filter_len (int): length of the filters in frames
            n_filters (int): number of filters
            strides: strides for the convolution operation (see tf.nn.conv2d)
            sigma: non-linearity to apply to the result (e.g., tf.nn.relu
                or tf.nn.tanh).
        """
        x_A = self.subnet_A[-1] 
        x_B = self.subnet_B[-1]
        assert np.all(x_A.get_shape() == x_A.get_shape())

        shape_x = x_A.get_shape()
        n_frames, n_bins, n_channels = [int(dim) for dim in shape_x[1:]]
        
        if not n_bins == 1:
            raise ValueError('dimension 2 should be 1 (in current implementation)')

        W = self.weight_variable([filter_len, 1, 1, n_filters])
        
        for subnet in [self.subnet_A, self.subnet_B]:
            x = subnet[-1]

            # transpose (None, n_frames, 1, N)
            #        -> (None, n_frames, N, 1)
            x_T = tf.transpose(x, perm=[0,1,3,2])

            # conv (None, n_frames, N, 1)
            #   -> (None, n_frames, N, n_filters)
            x_conv = tf.nn.conv2d(x_T, W, strides=strides, padding='SAME')

            # tile (None, n_frames, N, 1)
            #   -> (None, n_frames, N, n_filters)
            x_tile = tf.tile(x_T, [1,1,1,n_filters])

            # transpose (None, n_frames, N, n_filters)
            #        -> (None, n_filters, n_frames, N)
            x_conv_T = tf.transpose(x_conv, perm=[0, 3, 1, 2])
            x_tile_T = tf.transpose(x_tile, perm=[0, 3, 1, 2])

            # matmul (None, n_filters, n_frames, N) x idem
            #     -> (None, n_filters, N**2)
            matmul = tf.batch_matmul(x_conv_T, x_tile_T, adj_x=True)

            # flatten (None, n_filters, N**2)
            #      -> (None, n_filters * N**2)
            matmul_flat = tf.reshape(matmul, [-1, n_filters * n_channels**2])
            subnet.append(sigma(matmul_flat))

    def weight_variable(self, shape, weight_scale=0.1):
        """Return Tensorflow variable with a given dimension,
            initialized with tf.truncated_normal with standard
            deviation weight_scale.

        Args:
            shape (list): the variable's dimensions as a list
            weight_scale (float): standard_deviation of initial
                values

        Returns:
            tf.Variable: the variable
        """
        initial = tf.truncated_normal(shape, stddev=weight_scale)
        return tf.Variable(initial, name='a_weight')

    def bias_variable(self, shape, weight_scale=0.1):
        """Return Tensorflow variable with a given dimension,
            initialized as a tf.constant equal to
            parameter weight_scale.

        Args:
            shape (list): the variable's dimensions as a list
            weight_scale (float): initial value

        Returns:
            tf.Variable: the variable
        """
        initial = tf.constant(weight_scale, shape=shape)
        return tf.Variable(initial, name='a_bias')

    # def n_layers(self):
    #     assert len(self.subnet_A) == len(self.subnet_B)
    #     return len(self.subnet_A)

    # def remove_layers(self, n_keep):
    #     self.subnet_A = self.subnet_A[:n_keep+1]
    #     self.subnet_B = self.subnet_A[:n_keep+1]

    def loss(self, m=10, alpha=1):
        """Return loss function for training butterfly networks as a tensor.

        Minize pair distances while maximizing non-pair distances smaller
            than `m`.

        Returns:
            tf.Tensor: butterfly loss as a tensor.
        """
        y_A, y_B = self.subnet_A[-1], self.subnet_B[-1]
        squared_dists = tf.reduce_sum(tf.square(y_A - y_B),
                                      reduction_indices=1)

        pair_errors = squared_dists
        non_pair_errors = tf.square(tf.maximum(0.0, m - tf.sqrt(squared_dists)))

        pair_loss = tf.reduce_mean(self.is_cover * pair_errors, name='pair_loss')
        non_pair_loss = tf.reduce_mean((1 - self.is_cover) * non_pair_errors, name='non_pair_loss')
        total_loss = tf.add(pair_loss, alpha * non_pair_loss, name='loss')

        return total_loss, pair_loss, non_pair_loss

    def bhattacharyya(self):
        """Approximate bhattacharyya distance between cover and non-cover distances.
        
        Similar to Mahalanobis distance, but for distributions with different variances.
        Assumes normality, hence approximate.

        Returns:
            tf.Tensor: bhattacharyya distance between distributions of the cover
                and non-cover pairs' distances.
            tf.Tensor: mean cover pair distance
            tf.Tensor: mean non-cover pair distance
        """
        y_A, y_B = self.subnet_A[-1], self.subnet_B[-1]
        squared_dists = tf.reduce_sum(tf.square(y_A - y_B),
                                      reduction_indices=1, )
        
        cover_pairs = tf.where(tf.equal(self.is_cover, tf.ones_like(self.is_cover)))
        non_cover_pairs = tf.where(tf.equal(self.is_cover, tf.zeros_like(self.is_cover)))

        pair_dists = tf.sqrt(tf.gather(squared_dists, tf.reshape(cover_pairs, [-1])))
        non_pair_dists = tf.sqrt(tf.gather(squared_dists, tf.reshape(non_cover_pairs, [-1])))
        
        mu_pairs, sigma2_pairs = tf.nn.moments(pair_dists, axes=[0], name='d_pairs')
        mu_non_pairs, sigma2_non_pairs = tf.nn.moments(non_pair_dists, axes=[0], name='d_non_pairs')

        bhatt = tf.add( 0.25 * tf.log(0.25 * (sigma2_pairs/sigma2_non_pairs + sigma2_non_pairs/sigma2_pairs + 2)),
                  0.25 * (mu_pairs - mu_non_pairs)**2 / (sigma2_pairs + sigma2_non_pairs), name='bhatt')
        return bhatt, mu_pairs, mu_non_pairs

    def train_step(self, loss, learning_rate=3e-4):
        if loss is None:
            # only if not later needed for logging
            loss, _, _ = self.loss()
        adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return adam

    def log_errors(self, session, train_batch, test_batch, metrics,
                   log_every=1, verbose=True):
        """Compute train and test metrics and add to training log `train_log`.

        Args:
            session (tf.Session): session in which to run the metric evaluation
            train_batch (tuple): batch of input training data
                (x_A, x_B, is_cover)
            test_batch (tuple): 
        """
        def __strip__(metric_name, strip_slash=True, strip_colon=True):
            # strip everything after '/' and/or ':' from var names
            stripped = metric_name
            if strip_slash:
                stripped = metric_name.split('/')[0]
            if strip_colon:
                stripped = stripped.split(':')[0]
            return stripped

        if self.log_count % log_every == 0:

            train_metric_names = ['TR.' + __strip__(metric.name) for metric in metrics]
            test_metric_names = ['TE.' + __strip__(metric.name) for metric in metrics]

            if self.train_log is None:
                col_names = train_metric_names + test_metric_names
                self.train_log = pd.DataFrame(columns=col_names)
            
            # train and test feeds
            train_feed = {self.x_A:train_batch[0], self.x_B:train_batch[1],
                          self.is_cover: train_batch[2]}
            test_feed = {self.x_A:test_batch[0], self.x_B:test_batch[1],
                         self.is_cover: test_batch[2]}
            
            # compute and log metrics
            train_metrics = session.run(metrics, feed_dict=train_feed)
            self.train_log.loc[self.log_count, train_metric_names] = train_metrics

            test_metrics = session.run(metrics, feed_dict=test_feed)
            self.train_log.loc[self.log_count, test_metric_names] = test_metrics

            # optionally print last row
            if verbose:
                print(self.train_log[-1:], '\n')

        self.log_count += 1

    def fingerprint(self, chroma, n_patches=8, patch_len=64):
        n_frames, n_bins = chroma.shape
        if not n_frames == n_patches * patch_len:
            chroma = paired_data.patchwork(chroma, n_patches=n_patches,
                                           patch_len=patch_len)
        fps = []
        for i in range(12):
            patchwork_transposed = np.roll(patchwork, -i, axis=1)
            patchwork_tensorshaped = patchwork_transposed.reshape((1, n_patches*patch_len, 12))
            network_out = self.subnet_A[-1]
            fp = network_out.eval(feed_dict={x_A_in : patchwork_tensorshaped})
            fps.append(fp.flatten())
        return fps


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