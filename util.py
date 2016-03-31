#!/usr/bin/env python

"""Cover ID auxiliary methods.

Mainly data handling that is not specific to the dataset.
"""

from __future__ import division, print_function

import numpy as np
from sklearn.cross_validation import train_test_split


def split_train_test_validation(clique_dict, ratio=(50,20,30),
                               random_state=1988):
    """Split cliques into train, test and validation dataset.
    
    Args:
        clique_dict (dict): clique dataset as a dict with clique names
            as keys and lists of song URI's as values
        ratio (tuple): length-3 tuple speficying the ratio of train,
            test and validation set size.
    Returns:
        tuple: len-3 tuple containing the train, test, and validation
            partioning of the clique dictionary
    """
    # scale ratios to sum to 1
    ratio = ratio / np.sum(ratio)
    
    clique_names = clique_dict.keys()
    
    # make validation set
    train_test, val = train_test_split(clique_names,
                                       test_size=ratio[-1],
                                       random_state=random_state)
    val_cliques = {clique_name: clique_dict[clique_name] for
                   clique_name in val}

    # rescale (ratio[0], ratio[1]) to sum to 1
    ratio = ratio[:-1] / np.sum(ratio[:-1])
    
    # make train & test set
    train, test =  train_test_split(train_test,
                                    test_size=ratio[1],
                                    random_state=random_state)
    train_cliques = {clique_name: clique_dict[clique_name] for
                     clique_name in train}
    test_cliques = {clique_name: clique_dict[clique_name] for
                     clique_name in test}
    
    return train_cliques, test_cliques, val_cliques


def uris_from_clique_dict(clique_dict):
    """Return list of all uris in a clique dictionary.

    Args:
        clique_dict (dict): dictionary of clique names (keys) each
            pointing to a list or uris

    Returns:
        list: list of all uris in the dictionary
    """
    
    uris = [uri for clique in clique_dict for uri in clique_dict[clique]]

    return uris

