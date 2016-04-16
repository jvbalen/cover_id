#!/usr/bin/env python

"""Tests for `evaluation.py`."""

from __future__ import division, print_function

import unittest
import numpy as np
from itertools import combinations

from cover_id import SHS_data
from cover_id import evaluation


class Test_evaluate_query(unittest.TestCase):

    def setUp(self):
        self.query = 'TR1'
        self.correct = ['TR2', 'TR3']

    def test_zero(self):
        """Do bad results return ap = 0?"""
        retrieved = ['TR4', 'TR5', 'TR6', 'TR7']
        res = evaluation.evaluate_query(self.query, retrieved, self.correct)
        self.assertEqual(res['ap'], 0)
        self.assertEqual(res['p1'], 0)
        self.assertEqual(res['r5'], 0)

    def test_one(self):
        """Do perfect results return ap = 1"""
        # retrieved = ['TR2', 'TR3', 'TR4', 'TR5']
        retrieved = ['TR2', 'TR3', 'TR4', 'TR5']
        res = evaluation.evaluate_query(self.query, retrieved, self.correct)
        self.assertEqual(res['ap'], 1)
        self.assertEqual(res['p1'], 1)
        self.assertEqual(res['r5'], 1)

    def test_half(self):
        """Do ap = 0.5 results return correct ap?"""
        retrieved = ['TR4', 'TR2', 'TR5', 'TR3']
        res = evaluation.evaluate_query(self.query, retrieved, self.correct)
        self.assertEqual(res['ap'], 0.5)
        self.assertEqual(res['p1'], 0)
        self.assertEqual(res['r5'], 1)

    def test_half_real_data(self):
        query = 'TRKVQBG128F427E511'
        retrieved = ['TRQFPOO128F92E27A4', 'TRTTHWG128F92F6F57',  # these two are incorrect
                     'TRQUXMZ128F92D16D2', 'TRQJONS128F428D06E']  # these two are correct
        correct = ['TRQUXMZ128F92D16D2', 'TRQJONS128F428D06E']
        res = evaluation.evaluate_query(query, retrieved, correct)
        self.assertEqual(res['ap'], np.mean([1.0/3, 2.0/4]))
        self.assertEqual(res['p1'], 0)
        self.assertEqual(res['r5'], 1)


if __name__ == '__main__':
    unittest.main()