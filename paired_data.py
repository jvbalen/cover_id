#!/usr/bin/env python

"""Data preparation for feature learning on cover songs"""

from __future__ import division, print_function

import numpy as np
from numpy.random import choice
from itertools import combinations


def dataset_of_pairs(clique_dict, chroma_dict, n_patches=4, patch_len=64):
    """Construct a dataset of cover and non-cover chroma feature pairs.

    The `patchwork` function from this module is used to re-arrange chroma
        into same-size features for all songs.

    Args:
        clique_dict (dict): clique dataset as a dict with clique names
            as keys and lists of song URI's as values
        chroma_dict (dict): dictionary of uri (str) - chroma feature
            (2d-array) mappings, as returned by SHS_data.preload_chroma

    Returns:
        3d-array: features for first song in pair
        3d-array: features for second song in pair
        1d-array: boolean array indicating if pair is a cover pair
        list: list of (uri_1, uri_2) tuples for each pair
    """
    
    pairs, non_pairs = get_pairs(clique_dict)
    
    X_1, X_2 = [], []
    is_cover, pair_uris = [], []
    for pair, non_pair in zip(pairs, non_pairs):
        
        # pair
        chroma_1 = chroma_dict[pair[0]]
        chroma_2 = chroma_dict[pair[1]]
        patchwork_1 = patchwork(chroma_1, n_patches=n_patches, patch_len=patch_len)
        patchwork_2 = patchwork(chroma_2, n_patches=n_patches, patch_len=patch_len)
        x_1, x_2 = align_pitch(patchwork_1, patchwork_2)
        X_1.append(x_1)
        X_2.append(x_2)
        is_cover.append(True)
        pair_uris.append(pair)
        
        # non-pair
        chroma_1 = chroma_dict[non_pair[0]]
        chroma_2 = chroma_dict[non_pair[1]]
        patchwork_1 = patchwork(chroma_1, n_patches=n_patches, patch_len=patch_len)
        patchwork_2 = patchwork(chroma_2, n_patches=n_patches, patch_len=patch_len)
        x_1, x_2 = align_pitch(patchwork_1, patchwork_2)
        X_1.append(x_1)
        X_2.append(x_2)
        is_cover.append(False)
        pair_uris.append(non_pair)

    Y = np.array(is_cover, dtype=float).reshape((-1,1))
    
    return np.array(X_1), np.array(X_2), Y, pair_uris


def get_pairs(clique_dict):
    """Get all pairs of cover songs in a clique dataset, and a sample of
        non-cover pairs of the same size.

    Args:
        clique_dict (dict): clique dataset as a dict with clique names
            as keys and lists of song URI's as values

    Returns:
        pairs (list): list of pairs (each a tuple)
    """
    pairs = []
    non_pairs = []
    for this_clique in clique_dict:

        # clique uris 
        clique_uris = clique_dict[this_clique]

        # non-clique uris
        other_cliques = [clique for clique in clique_dict if not clique == this_clique]
        non_clique_uris = [uri for clique in other_cliques for uri in clique_dict[clique]]
        
        # clique pairs
        clique_pairs = list(combinations(clique_uris, 2))

        # clique non-pairs = [some clique uri, some non-clique uri] x len(clique pairs)
        n_clique_pairs = len(clique_pairs)
        clique_sample = choice(clique_uris, n_clique_pairs, replace=True)
        non_clique_sample = choice(non_clique_uris, n_clique_pairs, replace=False)
        clique_non_pairs = zip(clique_sample, non_clique_sample)
        
        pairs.extend(clique_pairs)
        non_pairs.extend(clique_non_pairs)

    return pairs, non_pairs


def patchwork(chroma, n_patches=4, patch_len=64):
    """Re-arrange chroma features into a fixed length sequence of
        patches.

    Args:
        chroma (2d-array): 2d-array containing the chroma features

    Returns:
        2d-array: 2d-array containing the re-arranged chroma
            features.
    """
    min_len = patch_len + 1

    if len(chroma) <= min_len:
        n_repetitions = min_len // len(chroma) + 1
        chroma = np.tile(chroma, (n_repetitions, 1))

    last_patch = len(chroma) - patch_len
    hop_length =(last_patch) / (n_patches - 1)
    
    t_begin = np.round(np.arange(0, last_patch + hop_length, hop_length)).astype(int)

    patches = np.vstack([chroma[t:t+patch_len] for t in t_begin])

    return patches

                         
def align_pitch(chroma_1, chroma_2):
    """Align the pitch dimension of two chroma features by
        circularly shifting the second along its pitch axis until their
        pitch histograms are maximally (circularly) cross-correlated.

    Args:
        chroma_1 (2d-array): first chroma array
        chroma_2 (2d-array): second chroma array

    Returns:
        2d-array: first chroma_array (unchanged)
        2d-array: pitch-aligned second chroma array
    """
    
    def __circ_conv__(x,y):
        X = np.fft.fft(x)
        Y = np.fft.fft(y)
        return np.fft.ifft(X * Y).real
    
    def __circ_corr__(x, y):
        x_roll = np.roll(x, 1)
        y_rev = y[::-1]
        return __circ_conv__(x_roll, y_rev)

    pitch_histogram_1 = np.mean(chroma_1, axis=0)
    pitch_histogram_2 = np.mean(chroma_2, axis=0)
                        
    ref_pitch = np.argmax(__circ_corr__(pitch_histogram_2, pitch_histogram_1))
                     
    # transpose chroma_2
    chroma_2_aligned = np.roll(chroma_2, -ref_pitch, axis=1)
                         
    return chroma_1, chroma_2_aligned


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

