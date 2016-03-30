#!/usr/bin/env python

"""Cover ID experiments.

Main function is run_leave_one_out_experiment.

The primary data structures used here are:

- datasets of cliques (dict)
    Clique datasets are dictionaries with clique names as keys and
    lists of track URIs as values.

- databases of fingerprints (dict)
    Fingerprint databases are dictionaries with URIs as keys and
    fingerprints as values. See documentation of fingerprints module
    for details.
"""

from __future__ import division, print_function

import numpy as np
import scipy.spatial.distance as dist

import SHS_data
import evaluation


def run_leave_one_out_experiment(clique_dict, fp_function, print_every=10):
    """Run a leave-one-out experiment given a dataset and
        fingerprinting function.
    
    Args:
        clique_dict (dict): dataset of cliques, as a dictionary
            (keys are clique names, values are lists of uris).
        fp_funtion: fingerprinting function that returns a flat numpy array
            given a uri, see e.g., `fp_corr`.
        print_every (int): how often to print progress

    Returns:
        dict: dictionary of results, averaged over all queries.
            Includes the mean of each evaluation metric returned by
            evaluation.evaluate_query().
    """
    uris = uris_from_clique_dict(clique_dict)
    
    # compute fingerprints
    print('Computing fingerprints...')
    fingerprints = fingerprint_tracks(uris, fp_function, print_every)
    
    # initialize results dictionary
    results_by_query = dict.fromkeys(uris)
    
    # querying clique by clique for efficient evaluation
    print('Running queries...')
    for i, clique in enumerate(clique_dict):
        if np.mod(i+1, print_every) == 0:
            print('    Running queries for clique',
                  '{}/{}'.format(i+1,len(clique_dict)))
        
        clique_uris = clique_dict[clique]

        for query in clique_uris:
            
            # query fingerprint database
            ranked_candidates = query_database(fingerprints, query)
            
            # evaluate
            results_by_query[query] = evaluation.evaluate_query(query,
                                                                ranked_candidates,
                                                                clique_uris)
    # average metrics
    metrics = results_by_query.values()[0].keys()
    results = {'mean ' + metric :
               np.mean([results_by_query[uri][metric] for uri in uris])
               for metric in metrics}
    
    return results


def fingerprint_tracks(uris, fp_function, print_every=10):
    """Compute fingerprints for a collection of tracks.

    Args:
        uris (list): list of uris of the tracks to be fingerprinted.
        fp_funtion: fingerprinting function that returns a flat numpy
            array given a uri, see e.g., `fp_corr`.

    Returns:
        dict: dictionary of fingerprints, keys are uri, values
            are fingerprints, one per uri, as a 1d-array.
    """
    n_uris = len(uris)

    # database is a dictionary
    fp_database = dict.fromkeys(uris)

    for i, uri in enumerate(uris):

        # print progress sometimes
        if np.mod(i+1, print_every) == 0:
            print('    Fingerprinting track {}/{}...'.format(i+1, n_uris))
        
        chroma = SHS_data.read_chroma(uri)
        
        fp_database[uri] = fp_function(chroma)

    return fp_database


def query_database(fp_database, query_uri, dist_metric='cosine'):
    """Query fingerprint database.

    Query a dictionary of fingerprints, indexed by uri, given a query
        uri. 
        Dictionary may contain either 1 fingerprint (1d-array) or
        a list of fingerprints (each a 1d-array).
        Multiple fingerprints (say, N) are treated as transpositions:
        each is used once if the track is a query, while only one is
        used when a track is in the candidate set. Distances between
        query and candidates are min-pooled across the N candidate
        fingerprints.
    
    Args:
        database (dict): dictionary of uri - fingerprint pairs.
            If multiple fingerprints are given, each dict entry
            should contain a list of 1d-arrays.
        query uri (str): uri of the track to be used as query.
        dist_metric (str): valid scipy.spatial.distance metric.

    Returns:
        list: list of ranked candidate uris.
    """
    db = fp_database.copy()
    query_fp = db.pop(query_uri) 
    
    candidate_uris = db.keys()
    candidate_fps = db.values()

    # several fingerprints
    if type(query_fp) is list and len(query_fp[0].shape) == 1:

        # use only one fingerprint per candidate
        candidate_fps = [fp_allkeys[0] for fp_allkeys in candidate_fps]
        
        distances_all_keys = dist.cdist(query_fp, candidate_fps,
                                        metric=dist_metric)
        distances = np.min(distances_all_keys, axis=0)

    # one fingerprint
    elif len(query_fp.shape) == 1:
        distances = dist.cdist([query_fp], candidate_fps,
                                metric=dist_metric)[0]
    
    # weird shape
    else:
        raise TypeError('Fingerprint should be 1d-array or ' +
                        'a list of 1-d arrays.')
    
    ranked_uris = np.array(candidate_uris)[np.argsort(distances)]

    return ranked_uris



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
