#!/usr/bin/env python

"""Tests for `paired_data.py`."""

from __future__ import division, print_function

import unittest
import numpy as np

import SHS_data
import util

from cover_id import paired_data


class Test_get_pairs(unittest.TestCase):
    """Tests for `get_pairs'."""

    def setUp(self):
        ratio = (1,9,90)
        clique_dict, self.cliques_by_uri = SHS_data.read_cliques()
        self.train_cliques, _, _ = util.split_train_test_validation(clique_dict,
                                                                    ratio=ratio)

    def test_dimensions(self):
        """Are `pairs` and `non_pairs` the same length, and all 
            pairs and non-pairs length-2?
            And are all pairs (and no non-pairs) indeed from the same clique?
        """
        pairs, non_pairs = paired_data.get_pairs(self.train_cliques)

        self.assertEqual(len(pairs), len(non_pairs))

        self.assertTrue(np.all([len(pair) == 2 for pair in pairs]))
        self.assertTrue(np.all([len(non_pair) == 2 for non_pair in non_pairs]))

        self.assertTrue(np.all([self.cliques_by_uri[pair[0]] == 
                                 self.cliques_by_uri[pair[1]]
                                 for pair in pairs]))
        self.assertTrue(not np.any([self.cliques_by_uri[non_pair[0]] == 
                                     self.cliques_by_uri[non_pair[1]]
                                     for non_pair in non_pairs]))


class Test_patchwork(unittest.TestCase):
    """Tests for `patchwork'."""

    def setUp(self):
        ratio = (1,9,90)
        clique_dict, _ = SHS_data.read_cliques()
        train_cliques, _, _ = util.split_train_test_validation(clique_dict,
                                                               ratio=ratio)
        self.pairs, self.non_pairs = paired_data.get_pairs(train_cliques)

    def test_artificial_data(self):
        """Do the patchworks share first and last frame with chroma?
            And do they have the correct size?"""
        n_patches, patch_len = 3, 10

        for len_x in range(1, 2 * n_patches * patch_len):

            chroma = np.random.rand(len_x, 12)
            patches = paired_data.patchwork(chroma, n_patches=n_patches, patch_len=patch_len)

            self.assertTrue(np.allclose(patches[0], chroma[0]))
            self.assertTrue(np.allclose(patches[-1], chroma[-1]))

            self.assertEqual(len(patches), n_patches * patch_len, msg='len_x = {}'.format(len_x))

    def test_real_data(self):
        """Do the patchworks share first and last frame with chroma?
            Do patchworks for a given pair have the same size?
        """
        test_pair = self.pairs[0]
        chroma_1 = SHS_data.read_chroma(test_pair[0])
        chroma_2 = SHS_data.read_chroma(test_pair[1])
        patches_1 = paired_data.patchwork(chroma_1)
        patches_2 = paired_data.patchwork(chroma_2)
        
        test_non_pair = self.non_pairs[0]
        chroma_3 = SHS_data.read_chroma(test_non_pair[0])
        chroma_4 = SHS_data.read_chroma(test_non_pair[1])
        patches_3 = paired_data.patchwork(chroma_3)
        patches_4 = paired_data.patchwork(chroma_4)


        self.assertTrue(np.all(patches_1[0] == chroma_1[0]))
        self.assertTrue(np.all(patches_1[-1] == chroma_1[-1]))
        self.assertTrue(np.all(patches_2[0] == chroma_2[0]))
        self.assertTrue(np.all(patches_2[-1] == chroma_2[-1]))
        self.assertTrue(np.all(patches_3[0] == chroma_3[0]))
        self.assertTrue(np.all(patches_3[-1] == chroma_3[-1]))
        self.assertTrue(np.all(patches_4[0] == chroma_4[0]))
        self.assertTrue(np.all(patches_4[-1] == chroma_4[-1]))

        self.assertEqual(patches_1.shape, patches_2.shape)
        self.assertEqual(patches_3.shape, patches_4.shape)

    def test_TRYBFEO12903CED194(self):
        n_patches = 8
        patch_len = 64

        test_uri = 'TRYBFEO12903CED194'

        chroma = SHS_data.read_chroma(test_uri)
        patches = paired_data.patchwork(chroma, n_patches=n_patches, patch_len=patch_len)

        self.assertTrue(np.allclose(patches[0],chroma[0]))
        self.assertTrue(np.allclose(patches[-1], chroma[-1]))

        self.assertEqual(len(patches), n_patches * patch_len)


    def test_short_chroma(self):
        """Do the patchworks for a given pair have the same size?"""
        test_pair = self.pairs[0]
        chroma_1 = SHS_data.read_chroma(test_pair[0])
        chroma_2 = SHS_data.read_chroma(test_pair[1])

        n_patches = 3
        patch_len = min(len(chroma_1), len(chroma_2)) - 1

        patches_1 = paired_data.patchwork(chroma_1, n_patches=n_patches, patch_len=patch_len)
        patches_2 = paired_data.patchwork(chroma_2, n_patches=n_patches, patch_len=patch_len)

        self.assertEqual(patches_1.shape, patches_2.shape)


class Test_align_pitch(unittest.TestCase):
    """Tests for `align_pitch'."""

    def test_basis_vector(self):
        """Does the function re-align a rolled basis vector?"""
        e = np.array([[1,0,0,0,0,0],
                      [1,0,0,0,0,0]])
        n_roll = -3
        rolled = np.roll(e, n_roll, axis=1)
        _, aligned = paired_data.align_pitch(e, rolled)
        self.assertTrue(np.allclose(e, aligned))

    def test_diatonic(self):
        """Does the function re-align a transposed diatonic profile?"""
        diatonic = np.array([[1,0,1,0,1, 1,0,1,0,1,0,1],
                             [1,0,1,0,1, 1,0,1,0,1,0,1]])
        n_trans = 7
        transposed = np.roll(diatonic, n_trans, axis=1)
        _, aligned = paired_data.align_pitch(diatonic, transposed)
        self.assertTrue(np.allclose(diatonic, aligned))


class Test_dataset_of_pairs(unittest.TestCase):
    """Tests for `dataset_of_pairs`."""

    def setUp(self):
        ratio = (5,1,94)
        clique_dict, self.cliques_by_uri = SHS_data.read_cliques()
        self.train_cliques, _, _ = util.split_train_test_validation(clique_dict,
                                                                    ratio=ratio)
        train_uris = util.uris_from_clique_dict(self.train_cliques)

        print('\n(preloading chroma...)')
        self.chroma_dict = SHS_data.preload_chroma(train_uris)

    def test_output_shapes(self):
        """Are the dimensions of the dataset consistent?"""
        output = paired_data.dataset_of_pairs(self.train_cliques, self.chroma_dict)
        X_A, X_B, is_cover, pair_uris = output

        self.assertEqual(X_A.shape, X_B.shape)
        self.assertEqual(len(X_A.shape), 3)
        self.assertEqual(len(X_A), len(is_cover))
        self.assertEqual(len(X_A), len(pair_uris))

        for X_a, X_b in zip(X_A, X_B):
            self.assertEqual(X_a.shape, X_b.shape)

    def test_is_cover(self):
        """Are the target labels in is_cover balanced and consistent
            with pair uris?
        """
        output = paired_data.dataset_of_pairs(self.train_cliques, self.chroma_dict)
        X_A, X_B, is_cover, pair_uris = output

        verify_is_cover = [self.cliques_by_uri[uri_1] == self.cliques_by_uri[uri_2]
                           for uri_1, uri_2 in pair_uris]

        self.assertEqual(np.sum(is_cover==True), np.sum(is_cover==False))
        self.assertTrue(np.all(is_cover == verify_is_cover))



if __name__ == '__main__':
    unittest.main()