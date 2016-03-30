#!/usr/bin/env python

"""Tests for `main.py`."""

from __future__ import division, print_function

import unittest
import numpy as np

import SHS_data
import main
import util
import fingerprints as fp

class Test_fingerprint_tracks(unittest.TestCase):

    def setUp(self):
        test_ratio = (1,9,90)
        cliques_by_name, cliques_by_uri = SHS_data.read_cliques()
        train_cliques, _, _ = util.split_train_test_validation(cliques_by_name,
                                                                   ratio=test_ratio)
        self.train_uris = util.uris_from_clique_dict(train_cliques)

    def test_output_size(self):
        """Is len(database) == len(uris)?"""
        uris = self.train_uris
        database = main.fingerprint_tracks(uris, fp.fourier)
        self.assertEqual(len(database), len(uris))

    def test_output_size_one(self):
        """Is len(database) == len(uris) also for len(uri) = 1?"""
        uris = [self.train_uris[0]]
        database = main.fingerprint_tracks(uris, fp.fourier)
        self.assertEqual(len(database), len(uris))

    def test_output_size_zero(self):
        """Is len(database) == len(uris) also for len(uri) = 0?"""
        uris = []
        database = main.fingerprint_tracks(uris, fp.fourier)
        self.assertEqual(len(database), len(uris))

    def test_fingerprint_type(self):
        """Are all fingerprint either lists of 1d-arrays or 1d-arrays?"""
        database = main.fingerprint_tracks(self.train_uris, fp.fourier)
        for fingerprint in database.values():
            self.assertTrue((type(fingerprint) is list and len(fingerprint[0].shape) == 1)
                             or len(fingerprint.shape) == 1)


class Test_query_database(unittest.TestCase):

    def setUp(self):
        test_ratio = (1,9,90)
        cliques_by_name, cliques_by_uri = SHS_data.read_cliques()
        train_cliques, _, _ = util.split_train_test_validation(cliques_by_name,
                                                                   ratio=test_ratio)
        self.train_uris = util.uris_from_clique_dict(train_cliques)

    def test_query_in_results(self):
        """Does a query on a size-1 database return its only uri?"""
        query_uri = self.train_uris[0]
        database = main.fingerprint_tracks(self.train_uris, fp.fourier)
        ranked_uris = main.query_database(database, query_uri)
        self.assertTrue(not query_uri in ranked_uris)

    def test_database_size_two(self):
        """Does a query on a size-1 database return its only uri?"""
        db_uri, query_uri = self.train_uris[:2]
        database = main.fingerprint_tracks([db_uri, query_uri], fp.fourier)
        ranked_uris = main.query_database(database, query_uri)
        self.assertEqual(len(ranked_uris), 1)
        self.assertEqual(ranked_uris[0], db_uri)


if __name__ == '__main__':
    unittest.main()
        