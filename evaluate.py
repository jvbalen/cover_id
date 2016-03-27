#!/usr/bin/env python

"""Evaluate cover id experiment."""

import numpy as np
import os
from pandas import read_csv

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
    
    clique_dict = {}
    
    uris, ids = read_uris()
    cliques = np.empty(len(uris), dtype='S140')
    cliques[:] = ''
    
    clique_name = None
    clique_uris = []
    with open(clique_path) as f:
        for line in f.readlines():
            if line.startswith('%'):
                if not clique_name is None:
                    clique_dict[clique_name] = clique_uris
                clique_name = __strip_clique_name__(line)
                clique_uris = []
            elif not line.startswith('#'):
                uri = __strip_uri__(line)
                clique_uris.append(uri)
                try:
                    cliques[ids[uri]] = clique_name
                except KeyError:
                    # no id for uri 
                    pass
        clique_dict[clique_name] = clique_uris
    
    return clique_dict, cliques
