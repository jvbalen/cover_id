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


def cov(chroma, normalise='col'):
    """Fingerprint based on chroma and delta-chroma covariance matrix.

    After [1].

    Args:
        chroma (2d-array): 2d-array containing the chroma features.
        normalise (str): choose from 'mat' or 'col' to normalise by
            matrix or column norm, respectively.

    Returns:
        list: 12 fingerprints (1d-arrray), one for each key.

    [1] Kim, S., Unal, E., & Narayanan, S. (2008). Music fingerprint
        extraction for classical music cover song identification.
        IEEE Conference on Multimedia and Expo.
    """
    delta_chroma = np.diff(chroma, axis=0)
    chroma = chroma[:-1]
    
    fp_12 = []
    for i in range(12):
        # transpose
        delta_chroma_trans = np.roll(delta_chroma, i, axis=1)
        chroma_trans = np.roll(chroma, i, axis=1)

        # covariance matrix
        chroma_delta_chroma = np.hstack((chroma_trans, delta_chroma_trans))
        fp = np.cov(chroma_delta_chroma, rowvar=0)

        # normalise and flatten:
        if normalise == 'mat':
            # note: does not have any effect with dist_metric = 'cosine'
            fp = fp / np.linalg.norm(fp)
            upper = np.triu_indices_from(fp, k=0)
            fp = fp[upper]
        elif normalise == 'col':
            fp = np.array([col / np.linalg.norm(col) for col in fp])
            fp = fp.flatten()
        else:
            upper = np.triu_indices_from(fp, k=0)
            fp = fp[upper]

        fp_12.append(fp)

    return fp_12


def corr(chroma):
    """Chroma correlation coefficient fingerprints.

    After [1].

    Args:
        chroma (2d-array): 2d-array containing the chroma features.

    Returns:
        list: 12 fingerprints (1d-array), one for each key.

    [1] Van Balen, J., Bountouridis, D., Wiering, F., & Veltkamp, R.C.
        (2014). Cognition-inspired Descriptors for Scalable Cover Song
        Retrieval. In Proc. International Society for Music Information
        Retrieval Conference.
    """
    fp = np.corrcoef(chroma, rowvar=0)

    fp_12 = [np.roll(np.roll(fp, i, 0), i, 1) for i in range(12)]

    # flatten
    upper = np.triu_indices_from(fp, k=1)
    fp_12 = [fp[upper] for fp in fp_12]

    return fp_12


def fourier(chroma, n_ftm=64, hop_length=None, stretch=[1]):
    """2D Fourier Magnitude Coefficients.

    Fingerprint based on a 2D discrete Fourier transform of chroma
        features, as in [1].

    Args:
        chroma (2d-array): 2d-array containing the chroma features.
        n_ftm (int): length of the chunks or 'patches' from which the 
            Fourier transform is computed, in frames.
            Defaults to 64, originally 75 [1].
        hop_length (int):  distance in frames between consecutive
            patches. Defaults to n_ftm.
        stretch (list): time-stretch factors. Chroma is resampled with
            these factors beforing computing the fingerprint. One
            fingerprint is returned per time-stretch factor.
            Will prepend 1 if not in list.

    Returns:
        list: len(stretch) fingerprints, each a 1d-array of shape
            (12*n_ftm,).

    [1] Bertin-Mahieux, T., & Ellis, D. P. W. (2012). Large-Scale Cover
        Song Recognition Using The 2d Fourier Transform Magnitude.
        In Proc. International Society for Music Information Retrieval
        Conference.
    """
    if hop_length is None:
        hop_length = n_ftm

    # pad short chroma series with zero frames
    n_frames, n_bins = chroma.shape
    if n_frames < n_ftm:
        pad_frames = np.zeros((n_ftm - n_frames, n_bins))
        chroma = np.vstack((chroma, pad_frames))

    # prepend 1 in stretch
    stretch = list(stretch)  # make copy
    if 1 in stretch:
        stretch.remove(1)
    stretch.insert(0, 1)

    # power expansion
    chroma = chroma**2

    fp = []
    for s in stretch:
        # simple resampling
        t_stretch = np.arange(0, len(chroma), s).astype(int)
        chroma = chroma[t_stretch]

        # slice chroma to chunks
        chroma_chunks = [chroma[frame_i:frame_i+n_ftm, :]
                         for frame_i in range(0, len(chroma)+1 - n_ftm, hop_length)]
        
        # 2D Fourier transform magnitudes
        chunks_ftm = [np.abs(np.fft.fft2(chunk)) for chunk in chroma_chunks]

        # pool across chunks (originally: median)
        ftm = np.mean(chunks_ftm, axis=0)

        fp.append(ftm.flatten())
    
    if len(stretch) == 1:
        fp = fp[0]

    return fp
