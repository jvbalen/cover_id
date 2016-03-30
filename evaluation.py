#!/usr/bin/env python

"""Cover ID evaluation methods"""

from __future__ import division, print_function

import numpy as np
from warnings import warn


def evaluate_query(query, retrieved, correct_uris):
    """Evaluate retrieval results for a given query.
    
    Args:
        query (str): query URI
        retrieved (list): ordered list of top k retrieved documents
            (k can be anything)
        clique_uris (list): list of documents to be found.
    
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
        retrieved = list(retrieved)  # make copy
        retrieved.remove(query)
        warn('Current evaluation assumes \'leave-one-out\' paradigm.' +
             ' Query has been removed from retrieved songs.')
    
    if query in correct_uris:
        correct_uris = list(correct_uris)  # make copy
        correct_uris.remove(query)
        warn('Current evaluation assumes \'leave-one-out\' paradigm. ' +
             ' Query has been removed from correct uris.')
    
    # which retrieved documents are relevant (bool)?
    relevant = [r in correct_uris for r in retrieved]
    
    # ranks of relevant documents
    ranks = np.sort(np.where(relevant)[0]) + 1
    
    # average precision
    if len(ranks) > 0:
    	# precisions = 1/rank[0], 2/rank[1]...
		precisions = (np.arange(len(ranks)) + 1) / ranks
		average_precision = np.mean(precisions)
    else:
        average_precision = 0
    
    # precision at 1 and recall at 5
    p1 = __precision__(ranks, k=1)
    r5 = __recall__(ranks, correct_uris, k=5)
    
    return {'ap': average_precision, 'p1': p1, 'r5': r5}
