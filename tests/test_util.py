#!/usr/bin/env python

"""Tests for `util.py`."""

import unittest

import numpy as np
from itertools import combinations

from cover_id import util
from cover_id import SHS_data


class Test_split_train_test_validation(unittest.TestCase):
    """Tests for `split_train_test_validation'."""

    def setUp(self):
        self.cliques_by_name, self.cliques_by_uri = SHS_data.read_cliques()

    def test_some_ratio(self):
        """For some common set size ratio, are all sets non-empty?
        Do set sizes sum to the total number of cliques?
        Is there no overlap?
        """
        subsets = util.split_train_test_validation(self.cliques_by_name,
                                                       ratio=(50,20,30))
        subset_sizes = np.array([len(subset) for subset in subsets])

        # check non-empty
        self.assertTrue(np.all(subset_sizes > 0))

        # check total length
        self.assertEqual(np.sum(subset_sizes), len(self.cliques_by_name))

        # check overlap
        self.assertTrue(not check_clique_overlap(subsets))

    def test_no_validation(self):
        """For set sizes (x,y,0), is only the validation set empty?
        Do set sizes sum to the total number of cliques?
        Is there no overlap?
        """
        subsets = util.split_train_test_validation(self.cliques_by_name,
                                                       ratio=(70,30,0))
        subset_sizes = np.array([len(subset) for subset in subsets])

        # check empty / non-empty
        self.assertTrue(np.all(subset_sizes[:-1] > 0))
        self.assertTrue(subset_sizes[-1] == 0)

        # check total length
        self.assertEqual(np.sum(subset_sizes), len(self.cliques_by_name))

        # check overlap
        self.assertTrue(not check_clique_overlap(subsets))


def check_clique_overlap(dicts):
    """Check if any of a tuple or list of cliques show overlap."""

    sets = [set(dict_i.keys()) for dict_i in dicts]

    overlap = [len(set_i & set_j) for set_i, set_j in combinations(sets, 2)]

    return np.any(overlap)


if __name__ == '__main__':
    unittest.main()