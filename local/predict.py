#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import argparse
import os

import kaldi_io
import numpy as np
import onnxruntime


NUM_EMBEDDINGS_IN_ARK = 2000


def initialize_gpus(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def load_utt(ark, utt, position):
    with open(ark, 'rb') as f:
        f.seek(position - len(utt) - 1)
        ark_key = kaldi_io.read_key(f)
        assert ark_key == utt, 'Keys does not match: `{}` and `{}`.'.format(ark_key, utt)
        mat = kaldi_io.read_mat(f)
        return mat.transpose().copy()


def write_txt_vectors(path, data_dict):
    """ Write vectors file in text format.

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict):
            f.write('{}  [ {} ]{}'.format(name, ' '.join(str(x) for x in data_dict[name]), os.linesep))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='', type=str, help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--model', required=True, type=str, help='path to pretrained model')
    parser.add_argument('--ndim', required=True, type=int, default=64, help='dimensionality of features')
    parser.add_argument('--kaldi-data-dir', required=True, type=str, help='path to kaldi data directory')
    parser.add_argument('--emb-out-dir', required=True, type=str, help='output directory for storing embeddings')
    parser.add_argument('--continue-index', required=False, type=int, default=0, help='continue from last index')
    parser.add_argument('--batch_size', default=16, type=int)

    args = parser.parse_args()

    # gpu configuration
    initialize_gpus(args)

    assert os.path.isfile(args.model), f'Path to model `{args.model}` does not exist.'
    sess = onnxruntime.InferenceSession(args.model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    if not os.path.exists(args.emb_out_dir):
        os.makedirs(args.emb_out_dir)

    feats_path = os.path.join(args.kaldi_data_dir, 'feats.scp')
    assert os.path.exists(feats_path), f'Path `{feats_path}` does not exists.'

    emb_dict = {}
    with open(feats_path) as f:
        lines = f.readlines()
        num_lines = len(lines)
        for idx, line in enumerate(lines[args.continue_index:]):
            idx = idx + args.continue_index
            ark_idx = idx // NUM_EMBEDDINGS_IN_ARK
            utt, ark = line.split()
            ark, position = ark.split(':')
            # cut features to max 120 seconds
            fea = load_utt(ark, utt, int(position))
            fea = fea[:, :12000]

            # get embedding
            emb = sess.run([label_name], {input_name: fea[np.newaxis, :, :]})[0].squeeze()
            emb_dict[utt] = emb

            if idx % NUM_EMBEDDINGS_IN_ARK == NUM_EMBEDDINGS_IN_ARK - 1:
                print(f'Extracted {idx}/{num_lines}.')
                write_txt_vectors(os.path.join(args.emb_out_dir, 'xvector.{}.txt'.format(ark_idx)), emb_dict)
                emb_dict = {}

        if len(emb_dict) > 0:
            write_txt_vectors(os.path.join(args.emb_out_dir, 'xvector.{}.txt'.format(ark_idx)), emb_dict)
