#!/usr/bin/env python

"""Tests for `evaluate.py`."""

import unittest
import evaluate

import numpy as np
from types import StringType


class Test_read_uris(unittest.TestCase):
	"""Tests for `read_uris`."""

	def setUp(self):
		self.uris, self.ids = evaluate.read_uris()

	def test_not_empty(self):
		"""Are both returns non-empty?"""
		self.assertTrue(len(self.uris) > 0)
		self.assertTrue(len(self.ids) > 0)

	def test_same_length(self):
		"""Do both mappings have the same length?"""
		self.assertEqual(len(self.uris), len(self.ids))
	
	def test_id_to_uri_and_back(self):
		"""Does id > uri > id mapping return the original id?"""
		for test_id in range(len(self.ids)):
			self.assertEqual(self.ids[self.uris[test_id]], test_id)
		
	def test_uri_to_id_and_back(self):
		"""Does uri > id > uri mapping return the original uri?"""
		for uri in self.uris:
			self.assertEqual(self.uris[self.ids[uri]], uri)


class Test_read_cliques(unittest.TestCase):
	"""Tests for `read_cliques'."""

	def setUp(self):
		self.cliques_by_name, self.cliques_by_id = evaluate.read_cliques()
		self.uris, self.ids = evaluate.read_uris()

	def tests_cliques_not_empty(self):
		"""Are  lists non-empty for all cliques?"""
		for clique in self.cliques_by_id:
			self.assertTrue(len(clique) > 0)
	
	def tests_uris_not_empty(self):
		"""Are the uri lists non-empty for all cliques?"""
		for clique in self.cliques_by_name:
			self.assertTrue(len(self.cliques_by_name[clique]) > 0)

	def test_clique_to_id_and_back(self):
		"""Does clique > id > clique mapping return the original clique?"""
		for clique in self.cliques_by_name:
			invertable = [self.cliques_by_id[self.ids[uri]] == clique
				for uri in self.cliques_by_name[clique]]
			self.assertTrue(np.all(invertable), msg='clique = '.format(clique))

	def test_id_to_clique_and_back(self):
		"""Does id > clique > id mapping return the original id?"""
		for test_id in range(len(self.uris)):
			self.assertTrue(self.uris[test_id] in
				self.cliques_by_name[self.cliques_by_id[test_id]],
				msg='test_id = {}'.format(test_id))



if __name__ == '__main__':
    unittest.main()