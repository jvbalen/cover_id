#!/usr/bin/env python

"""I/O methods for the SHS dataset."""

from __future__ import division, print_function

import numpy as np
import os
from pandas import read_csv


# global vars
data_dir = '/Users/Jan/Documents/Work/Data/SHS_julien/'
chroma_dir = os.path.join(data_dir, 'chroma/')


def read_cliques(clique_file='shs_pruned.txt'):
    """Read database of uri-clique mappings.

    Args:
        clique_file (str): file name of the text file containing clique
            data. File is located in data_dir (global variable) and
            formatted as in the original SHS dataset:
            - rows starting with '#' are ignored
            - rows starting with '%' are clique names
            - any subsequent rows are URIs of songs in that clique
    Returns:
        cliques_by_name: dictionary containing a list of URIs per
            clique, indexed by clique name (str)
        cliques_by_uri: numpy array of clique names (str) indexed by
            uri (str)

    Note: we use Julien Osmalsky's pruned SHS dataset.
        This clique database (see link) includes no clique id's, only
        names. To disambiguate between cliques with the same name, we
        add a unique indentifier to the beginning of the clique name 
        (str) (unrelated to the SHS 'work' identifier distributed with
        the original SHS data).

    More information:
        http://www.montefiore.ulg.ac.be/~josmalskyj/code.php
    """
    clique_path = os.path.join(data_dir, clique_file)
    
    def __strip_clique_name__(clique_line):
        """Extract clique name from line of db text."""
        last_col = clique_line.split(',')[-1]
        prefix = '{}_'.format(len(cliques_by_name))
        return prefix + last_col.strip('% \n')
    
    def __strip_uri__(uri_line):
        """Extract uri from line of db text."""
        first_col = uri_line.split('<SEP>')[0]
        return first_col.strip(' \n')

    # initialize returns
    cliques_by_name = {}
    cliques_by_uri = {}

    # read clique file
    with open(clique_path) as f:

        clique_name = None
        clique_uris = []
        for line in f.readlines():

            # read clique name lines
            if line.startswith('%'):

                if not clique_name is None:
                    # write to dictionary
                    cliques_by_name[clique_name] = clique_uris

                clique_name = __strip_clique_name__(line)
                clique_uris = []

            # read clique URI lines
            elif not line.startswith('#'):
                uri = __strip_uri__(line)
                clique_uris.append(uri)
                cliques_by_uri[uri] = clique_name

        # write to dictionary
        cliques_by_name[clique_name] = clique_uris
    
    return cliques_by_name, cliques_by_uri


def read_chroma(uri, ext='.csv'):
    """Read chroma for a given URI.

    Args:
        uri (str): song uri

    Returns:
        2d-array: chroma array (rows are frames)
    """
    chroma_path = chroma_dir + uri + ext
    chroma_data = read_csv(chroma_path)
    return chroma_data.values


def preload_chroma(uris):
    """Load chroma features for a dataset into a dict.
        Not recommended for full SHS (~800Mb of numpy arrays).

    Args:
        uris (list): list of uris

    Returns:
        dict: dictionary of chroma arrays.
    """
    chroma_dict = dict.fromkeys(uris)
    for uri in uris:
        chroma_dict[uri] = read_chroma(uri)
    return chroma_dict


def read_uris():
    """Read database of uri-id mappings.
    (For compatibility with Julien Osmalsky's ids)
    
    Args:
        none
        
    Returns:
        uris (numpy array): numpy array of uri's (string) indexed by
            integer id (int)
        ids (pandas series): series of ids (int) indexed by uri
            (string)
    """
    db_file = os.path.join(data_dir, 'uri_id_mapping.csv')
    
    # uris as numpy array
    uris_series = read_csv(db_file, header=None, squeeze=True, index_col=1)
    uris = uris_series.sort_index().values
    
    # ids as pandas series
    ids = read_csv(db_file, header=None, squeeze=True, index_col=0)
    
    return uris, ids


