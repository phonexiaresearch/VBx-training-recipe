#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix


def read_txt_vectors_from_stdin():
    """ Read vectors in text format. This code expects correct format.

        Yields:
            Tuple(str, np.array): utt and vector
    """
    for line in sys.stdin:
        splitted_line = line.split()
        name = splitted_line[0]
        end_idx = splitted_line.index(']')
        vector_data = np.array([float(single_float) for single_float in splitted_line[2:end_idx]])
        yield name, vector_data


def write_txt_vector_to_stdout(utt, emb):
    """ Write vectors file in text format.

    Args:
        utt (str):
        emb (np.ndarray):
    """
    sys.stdout.write('{}  [ {} ]{}'.format(utt, ' '.join(str(x) for x in emb), os.linesep))


def l2_norm(vec_or_matrix):
    """ L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def get_class_means_between_and_within_covs(samples, classids, bias=True):
    """ Return class means and the between and shread within class covariance matrix.

    Args:
        samples (np.array): input data
        classids (np.array): identification of classes
        bias (bool): type of normalization, see np.cov for more information

    Returns:
        tuple: class means, between class covariance, within class covariance
    """
    nsamples, dim = samples.shape
    sample2class = classids_to_posteriors(classids)
    counts = np.array(sample2class.sum(1))
    means = np.array(sample2class.dot(samples)) / counts
    between_cov = (means - samples.mean(0)) * np.sqrt(counts)
    between_cov = between_cov.T.dot(between_cov) / nsamples
    within_cov = np.cov(samples.T, bias=bias) - between_cov
    return means, between_cov, within_cov


def classids_to_posteriors(classids, dtype='f4'):
    """ Transform classids into a sparse matrix of 1 and 0 that can be interpreted as posteriors.

    Args:
        classids (np.array): class identifications
        dtype (str): dtype o output sparse matrix

    Returns:
        scipy.sparse.coo_matrix: sparse matrix of 1 and 0 that can be interpreted as posteriors
    """
    nsamples = np.array(classids).squeeze().shape[0]
    class2post = coo_matrix((np.ones(nsamples, dtype), (classids, list(range(nsamples))))).tocsr()
    return class2post


def train(embeddings, labels, lda_dim, whiten):
    n_vectors, vector_dim = embeddings.shape

    unique_idxs = list(set(labels))
    idxs = np.array([unique_idxs.index(x) for x in labels])
    # subtract mean
    mean1 = np.mean(embeddings, axis=0)
    embeddings -= mean1

    # l2-norm
    embeddings = l2_norm(embeddings)

    # train LDA
    means, between_cov, within_cov = get_class_means_between_and_within_covs(embeddings, idxs)

    # Hossein's magic - within_cov has probably smaller numbers on training data than within_cov on test data
    # CW = CW + 0.01 * eye(size(CT, 1));
    within_cov = within_cov + 0.01 * np.diag(np.ones(within_cov.shape[0]))

    w, v = eigh(between_cov, within_cov)
    index = np.argsort(w)[::-1]
    v = v[:, index]

    lda = v[:, 0:lda_dim]
    if whiten:
        lda = lda.dot(np.diag(1. / np.sqrt(np.diag(lda.T.dot(between_cov + within_cov).dot(lda)))))

    embeddings = embeddings.dot(lda)

    # subtract mean again
    mean2 = np.mean(embeddings, axis=0)
    embeddings -= mean2

    # l2-norm again
    embeddings = l2_norm(embeddings)

    return embeddings, mean1, lda, mean2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--utt2spk', required=True, type=str, help='path to spk2utt')
    parser.add_argument('--output-h5', required=True, type=str, help='path to output h5 file')
    parser.add_argument('--lda-dim', required=False, default=128, type=int, help='LDA dimensionality')
    parser.add_argument('--whiten', required=False, default=False, action='store_true', help='do whitenning after LDA')

    args = parser.parse_args()

    utt2spk = {}
    with open(args.utt2spk) as f:
        for line in f:
            utt, spk = line.split()
            utt2spk[utt] = spk

    embeddings, labels, utts = [], [], []
    for utt, emb in read_txt_vectors_from_stdin():
        embeddings.append(emb)
        labels.append(utt2spk[utt])
        utts.append(utt)

    # train parameters
    embeddings, mean1, lda, mean2 = train(np.array(embeddings), labels, lda_dim=args.lda_dim, whiten=args.whiten)

    # dump them into h5 file
    with h5py.File(args.output_h5, 'w') as f:
        f.create_dataset('mean1', data=mean1)
        f.create_dataset('mean2', data=mean2)
        f.create_dataset('lda', data=lda)

    for utt, embedding in zip(utts, embeddings):
        write_txt_vector_to_stdout(utt, embedding)
