__author__ = 'jvanbalen'

import numpy as np


def evaluate(scores, ground_truth, metric='map'):
    n_queries = len(scores)
    assert n_queries == len(ground_truth)
    # following line: list with for every query an array with the ranks of the relevant documents
    ranks = [1 + rank(scores[q])[ground_truth[q] == 1] for q in range(n_queries)]
    if metric == 'ap':
        aps = [ap(r) for r in ranks]
        result = np.array(aps)
    if metric == 'map':
        aps = [ap(r) for r in ranks]
        result = np.mean(aps)
    if metric == 'p1':
        hits = [any(r == 1) for r in ranks]
        result = np.mean(hits)
    if metric == 'r1':
        result = recallk(ranks, groundtruth, 1)
    if metric == 'r5':
        result = recallk(ranks, groundtruth, 5)
    return np.round(result, 3)


def rank(x):
    temp = x.argsort()[::-1]
    ranks = np.zeros(len(x))
    ranks[temp] = np.arange(len(x))
    return ranks


def ap(ranks):
    if len(ranks) == 0:
        print 'Query with no relevant candidates associated... nan returned'
        return np.nan
    else:
        recall = np.arange(len(ranks)) + 1
        return np.mean(recall / ranks)


def recallk(ranks, groundtruth, k):
    hits = [sum(r <= k) for r in ranks]
    nrel = [sum(gt) for gt in groundtruth]
    return np.mean(np.array(hits) / np.array(nrel))