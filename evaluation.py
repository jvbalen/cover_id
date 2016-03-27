#!/usr/bin/env python

"""I/O methods for the SHS dataset."""

from __future__ import division, print_function

import numpy as np
from warnings import warn

import SHS_data


def evaluate_query(query, retrieved, correct_uris=None):
    """Evaluate retrieval results for a given query.
    
    Args:
        query (str): query URI
        retrieved (list): ordered list of top k retrieved documents
            (k can be anything)
        clique_uris (list): list of documents to be found. Set to None
            to look up using ID, pass a list to save i/o time.
    
    Returns:
        dict: dictionary of results with evaluation metrics as keys.
            currently implemented:
            - ap (average precision)
            - precision at 1
            - recall at 5
    """

    def __precision__(ranks, k=1):
    	return np.sum(np.array(ranks) <= k) / k

    def __recall__(ranks, correct_uris, k=5):
    	return np.sum(np.array(ranks) <= k) / len(correct_uris)
    
    if query in retrieved:
        warn('Query is returned among retrieved songs.' +
             'Current evaluation assumes \'leave-one-out\' paradigm.')

    if correct_uris is None:
        # read clique data
        cliques_by_name, cliques_by_uri = SHS_data.read_cliques()
        clique_name = cliques_by_uri[query]
        correct_uris = set(cliques_by_name[clique_name]) - set([query])

    print(correct_uris)
    
    # which retrieved documents are relevant?
    relevant = [r in correct_uris for r in retrieved]
    
    # ranks of relevant documents
    ranks = np.sort(np.where(relevant)[0]) + 1
    
    if len(ranks) == 0:
    	average_precision = 0
    else:
    	# precisions = 1/rank[0], 2/rank[1]...
		precisions = (np.arange(len(ranks)) + 1) / ranks
		average_precision = np.mean(precisions)
    
    # precision at 1 and recall at 5
    p1 = __precision__(ranks, k=1)
    r5 = __recall__(ranks, correct_uris, k=5)
    
    return {'ap': average_precision, 'p1': p1, 'r5': r5}


