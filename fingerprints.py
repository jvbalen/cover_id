#!/usr/bin/env python

"""Fingerprinting functions for large-scale cover ID.

All fingerprints are 1d-arrays, or lists of 1d-arrays,
    in which case it will be assumed the different 1d-arrays
    are key transpositions. Queries will be run for each of 
    the transpositions, but only for one array per candidate
    (the first / i=0).

"""

from __future__ import division, print_function

import numpy as np


def rand(chroma, n=3):
    """Random fingerprints (for baseline computation)."""
    return np.random.rand(3)

def corr(chroma):
    """Chroma correlation coefficient fingerprints.

    Returns one fingerprint for each key.

    See:

    """
    fp = np.corrcoef(chroma, rowvar=0)

    fp_12 = [np.roll(np.roll(fp, i, 0), i, 1) for i in range(12)]

    # flatten
    upper = np.triu_indices_from(fp, k=1)
    fp_12 = [fp[upper] for fp in fp_12]

    return fp_12


def cov(chroma):
    """Chroma covariance fingerprints.

    Returns one fingerprint for each key.

    """
    fp = np.cov(chroma, rowvar=0)
    
    fp_12 = [np.roll(np.roll(fp, i, 0), i, 1) for i in range(12)]

    # flatten
    upper = np.triu_indices_from(fp, k=1)
    fp_12 = [fp[upper] for fp in fp_12]

    return fp_12


def fourier(chroma, n_ftm=64, hop_length=64):
    """2D Fourier Magnitude Coefficients.

    Fingerprint based on a 2D discrete Fourier transform of chroma
        features.

    Args:
        chroma (2d-array): 2d-array containing the chroma features.
        n_ftm (int): length of the chunks or 'patches' from which the 
            Fourier transform is computed, in frames.
        hop_length (int):  distance in frames between consecutive
            patches.
    """
    
    # power expansion
    chroma = chroma**2

    # slice chroma to windows
    chroma_chunks = [chroma[frame_i:frame_i+n_ftm, :]
                     for frame_i in range(0, len(chroma) - n_ftm, hop_length)]
    
    # 2D Fourier transform magnitudes
    chunks_ftm = [np.abs(np.fft.fft2(chunk)) for chunk in chroma_chunks]

    fp = np.mean(chunks_ftm, axis=0)

    fp = fp.flatten()
    
    return fp