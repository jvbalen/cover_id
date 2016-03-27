#!/usr/bin/env python

"""I/O for the SHS dataset."""

from __future__ import division, print_function

import numpy as np
import os
from pandas import read_csv
from sklearn.cross_validation import train_test_split


# global vars
data_dir = '/Users/Jan/Documents/Work/Data/SHS_julien/'


def read_uris():
    """Read database of uri-id mappings.
    
    Args:
        none
        
    Returns:
        uris (numpy array): numpy array of uri's (string) indexed by
            integer id (int)
        ids (pandas series): series of ids (int) indexed by uri
            (string)
        
    Note: uris is not a pandas series as it takes 100x longer to
        do look-up in a series compared to an array.
    """
    db_file = os.path.join(data_dir, 'uri_id_mapping.csv')
    
    # uris as numpy array
    uris = read_csv(db_file, header=None, squeeze=True, index_col=1)
    uris = np.array([uris[i] for i in range(len(uris))])
    
    # ids as pandas series
    ids = read_csv(db_file, header=None, squeeze=True, index_col=0)
    
    return uris, ids


def read_cliques(clique_file='shs_pruned.txt'):
    """Read database of uri-clique mappings.

    Args:
        clique_file (str): file name of the text file containing clique
            data. File is formatted as in the original SHS dataset:
            - rows starting with '#' are ignored
            - rows starting with '%' are clique names
            - any subsequent rows are URIs of songs in that clique
    Returns:
        clique_dict: dictionary containing a list of URIs per clique,
            indexed by clique name (str)
        cliques: numpy array of clique names (str) indexed by id (int)

    Note: we use Julien Osmalsky's pruned SHS dataset.
        This clique database (see link) includes no clique id's, only names.
        To disambiguate between cliques with the same name, we add a unique
        indentifier to the beginning of the clique name (str) (unrelated to
        the SHS 'work' identifier distributed with the original SHS data).

    See:
        pruned SHS dataset:
            http://www.montefiore.ulg.ac.be/~josmalskyj/files/shs_pruned.txt
        more information:
            http://www.montefiore.ulg.ac.be/~josmalskyj/code.php
    """
    clique_path = os.path.join(data_dir, clique_file)
    
    def __strip_clique_name__(clique_line):
        """Extract clique name from line of db text."""
        last_col = clique_line.split(',')[-1]
        prefix = '{}_'.format(len(clique_dict))
        return prefix + last_col.strip('% \n')
    
    def __strip_uri__(uri_line):
        """Extract uri from line of db text."""
        first_col = uri_line.split('<SEP>')[0]
        return first_col.strip(' \n')

    # initialize output dictionary
    clique_dict = {}

    # initialize output array
    uris, ids = read_uris()
    cliques = np.empty(len(uris), dtype='S140')
    cliques[:] = ''
    
    # initialize loop vars
    clique_name = None
    clique_uris = []

    # read clique file
    with open(clique_path) as f:

        for line in f.readlines():

            # read clique name lines
            if line.startswith('%'):

                # write to dictionary
                if not clique_name is None:
                    clique_dict[clique_name] = clique_uris

                clique_name = __strip_clique_name__(line)
                clique_uris = []

            # read clique URI lines
            elif not line.startswith('#'):
                uri = __strip_uri__(line)
                clique_uris.append(uri)

                try:
                    cliques[ids[uri]] = clique_name
                except KeyError:
                    # no id for uri 
                    pass

        # write to dictionary
        clique_dict[clique_name] = clique_uris
    
    return clique_dict, cliques


def split_train_test_validation(clique_dict, ratio=(50,20,30),
                               random_state=1988):
    """Split cliques into train, test and validation dataset.
    
    Args:
        clique_dict: dictionary with clique names as keys and lists of
            song URI's as values.
        ratio (tuple): length-3 tuple speficying the ratio of train,
            test and validation set size.
    Returns:
        tuple: len-3 tuple containing the train, test, and validation
            partioning of the clique dictionary
        tuple: len-3 tuple containing the list of ids relating to
            train, test and validation datasets
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
    
    # make train & test set
    train, test =  train_test_split(train_test,
                                    test_size=ratio[1],
                                    random_state=random_state)
    train_cliques = {clique_name: clique_dict[clique_name] for
                     clique_name in train}
    test_cliques = {clique_name: clique_dict[clique_name] for
                     clique_name in test}
    
    return train_cliques, test_cliques, val_cliques