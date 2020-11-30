#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Created Time:   2019-03-12
# @Author: Shuai Wang
# @Email: wsstriving@gmail.com
# @Last Modified Time: 2019-05-30

import random
import math
import torch
import os
import copy
import numpy as np

def split_scp(feat_scp, cachesize, file_dir, shuffle=False, regen=False):
    """
    :param feat_scp: The input feat.scp with too many lines to load
    :param cachesize: Split the file to sub_files with cachesize lines
    :param file_dir: tmp file directory to store sub_files
    :param shuffle: Shuffle the scp file before splitting
    :return: number of small files, small file paths
    """
    small_file = None
    small_file_names = []
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(feat_scp) as big_file:
        lines = big_file.readlines()
        if shuffle:
            random.shuffle(lines)

        for idx, line in enumerate(lines):
            if idx % cachesize == 0:
                if small_file:
                    small_file.close()
                small_feat_scp = file_dir + '/subfile_{}'.format(int(idx/cachesize)+1)
                small_file_names.append(small_feat_scp)
                if regen:
                    small_file = open(small_feat_scp, "w")
            if regen:
                small_file.write(line)
        if small_file:
            small_file.close()
    return int(math.ceil((idx + 1) * 1.0 / cachesize)), small_file_names

def validate_path(dir_name):
    """
    :param dir_name: Create the directory if it doesn't exist
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name is not ''):
        os.makedirs(dir_name)

def save_checkpoint_statdict(model_to_save, save_dir, is_best, epoch):
    """
    Save the stat_dict file, need the model difination when
    reloading models, only save the stat_dict such as weights
    Save space on disk
    """
    filename = os.path.join(save_dir, 'iter' + str(epoch) + '.sdict')
    validate_path(filename)
    model = copy.deepcopy(model_to_save)
    model = model.to(torch.device("cpu"))
    while isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.sdict'))


def save_checkpoint_model(model_to_save, save_dir, is_best, epoch, cache_idx):
    """
    Save the whole model file
    The model definition is also contained in the saved model
    """
    filename = os.path.join(save_dir, 'Epoch' + str(epoch) + "_iter" + str(cache_idx) + '.mdl')
    validate_path(filename)
    # Use loop since no idea why there may be layers of DataParallel wrapper
    model = copy.deepcopy(model_to_save)
    model = model.to(torch.device("cpu"))
    while isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model, filename)
    if is_best:
        torch.save(model, os.path.join(save_dir, 'model_best.mdl'))

def save_checkpoint_model_list(model_to_save, save_dir, is_best, epoch, cache_idx):
    """
    Save the whole model file
    The model definition is also contained in the saved model
    """
    filename = os.path.join(save_dir, 'Epoch' + str(epoch) + "_iter" + str(cache_idx) + '.mdl')
    validate_path(filename)
    # Use loop since no idea why there may be layers of DataParallel wrapper
    dumpmdl = []
    for mdl in model_to_save:
        model = copy.deepcopy(mdl)
        model = model.to(torch.device("cpu"))
        while isinstance(model, torch.nn.DataParallel):
            model = model.module
        dumpmdl.append(model)
    torch.save(dumpmdl, filename)
    if is_best:
        torch.save(dumpmdl, os.path.join(save_dir, 'model_best.mdl'))


# import from https://github.com/zcaceres/spec_augment
def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned

# import from https://github.com/zcaceres/spec_augment
def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
    return cloned

def time_mask_np(spec, T=40, num_masks=1, replace_with_zero=False, p=0.2):
    '''
        spec: np array, shape of (time_Step, feat_dim)
    '''
    cloned = spec.copy()
    len_spectro = cloned.shape[0]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, max(1, len_spectro - t))

        # avoids randrange error if values are equal and range is empty
        # Not useful if the input is of fixed length, T will be enough
        if (t_zero == t_zero + t) or (t > p*len_spectro): continue

        # Q: why sample again?
        # mask_end = random.randrange(t_zero, t_zero + t)
        mask_end =  t_zero + t
        if (replace_with_zero): cloned[t_zero:mask_end, :] = 0
        else: cloned[t_zero:mask_end, :] = cloned.mean()
    return cloned

# import from https://github.com/zcaceres/spec_augment, Modified
def freq_mask_np(spec, F=15, num_masks=1, replace_with_zero=False):
    '''
        spec: np array, shape of (time_Step, feat_dim)
    '''
    cloned = spec.copy()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): continue

        # Q: why sample again?
        # mask_end = random.randrange(f_zero, f_zero + f)
        mask_end = f_zero + f
        if (replace_with_zero): cloned[:, f_zero:mask_end] = 0
        else: cloned[:, f_zero:mask_end] = cloned.mean()

    return cloned
