#!/usr/bin/env python

"""Tests for `SHS_data.py`."""

import unittest

import numpy as np

from cover_id import SHS_data


class Test_read_cliques(unittest.TestCase):
	"""Tests for `read_cliques'."""

	def setUp(self):
		self.cliques_by_name, self.cliques_by_uri = SHS_data.read_cliques()
		self.uris, self.ids = SHS_data.read_uris()

	def tests_cliques_not_empty(self):
		"""Are clique names in clique_by_uri valid?"""
		for clique_name in self.cliques_by_uri.values():
			self.assertTrue(type(clique_name) is str)
			self.assertTrue(len(clique_name) > 0)
	
	def tests_uris_not_empty(self):
		"""Are the uri lists length-2 or more for all cliques?"""
		for clique in self.cliques_by_name:

			self.assertTrue(len(self.cliques_by_name[clique]) > 1)

	def test_clique_to_id_and_back(self):
		"""Does clique > id > clique mapping return the original clique?"""
		for clique in self.cliques_by_name:

			point_back = [self.cliques_by_uri[uri] == clique
				for uri in self.cliques_by_name[clique]]

			self.assertTrue(np.all(point_back), msg='clique = ' + clique)

	def test_uri_to_clique_and_back(self):
		"""Does uri > clique > uri mapping return the original uri?"""
		for test_uri in self.uris:

			self.assertTrue(test_uri in
				self.cliques_by_name[self.cliques_by_uri[test_uri]],
				msg='test_uri = {}'.format(test_uri))


class Test_read_uris(unittest.TestCase):
	"""Tests for `read_uris`."""

	def setUp(self):
		self.uris, self.ids = SHS_data.read_uris()

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



if __name__ == '__main__':
    unittest.main()