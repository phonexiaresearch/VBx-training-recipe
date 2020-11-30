#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Created Time:   2019-03-12
# @Author: Shuai Wang
# @Email: wsstriving@gmail.com
# @Last Modified Time: 2019-06-03


import numpy as np
from torch.utils.data import Dataset

import utils as kaldi_io


class KaldiDatasetFrame(object):
    """
    Generate the frame-wise data-target pair
    """
    def __init__(self, feat_scp, stream, label_dict, global_mean, global_std):
        super(KaldiDatasetFrame, self).__init__()
        print(feat_scp)
        feat_dict = {key: mat for key, mat in kaldi_io.read_mat_ark(stream.format(feat_scp))}
        # Python3 return dict_value instead of a list
        self.feats = list(feat_dict.values())
        self.labels = [label_dict[key] for key in feat_dict.keys()]

        self.feats = np.concatenate(self.feats)

        if global_mean is not None and global_std is not None:
            self.feats = (self.feats - global_mean) / global_std
        self.feats = self.feats.astype(np.float32)
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

    def __len__(self):
        return len(self.feats)

class KaldiDatasetUtt_PhnAll(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1 !
    FOR loading multiple labels, eg. phonemes for each frame
    DESIGNED for TDNN !!!!
    """
    def __init__(self, feat_scp, ali_dict, stream, global_mean=None, global_std=None):
        super(KaldiDatasetUtt_PhnAll, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(stream.format(feat_scp))
        self.ali_dict = ali_dict
        self.global_mean = global_mean
        self.global_std = global_std
        count = 0
        with open(feat_scp) as in_file:
            for line in in_file:
                count += 1
        self.length = count

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        tokens = name.split("-")
        spk_id = int(tokens[-1])
        utt_len = int(tokens[-2])
        begin_idx = int(tokens[-3])

        # print (begin_idx, utt_len, spk_id)

        if tokens[-4] == "noise" or tokens[-4] == "music" or tokens[-4] == "babble" or tokens[-4] == "reverb":
            orig_utt_name = "-".join(tokens[:-4])
        else:
            orig_utt_name = "-".join(tokens[:-3])
        # orig_utt_name = orig_utt_name.split("%")[-1]
        # print (self.ali_dict)
        frm_phn_label = self.ali_dict[orig_utt_name][begin_idx+7: begin_idx + utt_len -7]
        utt_phn_label = np.histogram(frm_phn_label, bins=range(168))[0]/len(frm_phn_label)

        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean) / self.global_std
        return feat, name, [spk_id, utt_phn_label, frm_phn_label]

    def __len__(self):
        return self.length

class KaldiDatasetUtt_PhnDist(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1 !
    FOR loading multiple labels, eg. phonemes will be aggrated to be utterances distribution
    DESIGNED for TDNN !!!!
    """
    def __init__(self, feat_scp, ali_dict, stream, global_mean=None, global_std=None):
        super(KaldiDatasetUtt_PhnDist, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(stream.format(feat_scp))
        self.ali_dict = ali_dict
        self.global_mean = global_mean
        self.global_std = global_std
        count = 0
        with open(feat_scp) as in_file:
            for line in in_file:
                count += 1
        self.length = count

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        tokens = name.split("-")
        spk_id = int(tokens[-1])
        utt_len = int(tokens[-2])
        begin_idx = int(tokens[-3])

        # print (begin_idx, utt_len, spk_id)

        if tokens[-4] == "noise" or tokens[-4] == "music" or tokens[-4] == "babble" or tokens[-4] == "reverb":
            orig_utt_name = "-".join(tokens[:-4])
        else:
            orig_utt_name = "-".join(tokens[:-3])
        # orig_utt_name = orig_utt_name.split("%")[-1]
        # print (self.ali_dict)
        phn_label = self.ali_dict[orig_utt_name][begin_idx+7: begin_idx + utt_len -7]
        # phn_label = [list(phn_label).count(i) for i in range(167)]
        phn_label = np.histogram(phn_label, bins=range(168))[0]/len(phn_label)

        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean) / self.global_std
        return feat, name, [spk_id, phn_label]

    def __len__(self):
        return self.length

class KaldiDatasetUtt_Phn(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1 !
    FOR loading multiple labels, eg. phonemes for each frame
    DESIGNED for TDNN !!!!
    """
    def __init__(self, feat_scp, ali_dict, stream, global_mean=None, global_std=None):
        super(KaldiDatasetUtt_Phn, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(stream.format(feat_scp))
        self.ali_dict = ali_dict
        self.global_mean = global_mean
        self.global_std = global_std
        count = 0
        with open(feat_scp) as in_file:
            for line in in_file:
                count += 1
        self.length = count

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        tokens = name.split("-")
        spk_id = int(tokens[-1])
        utt_len = int(tokens[-2])
        begin_idx = int(tokens[-3])

        # print (begin_idx, utt_len, spk_id)

        if tokens[-4] == "noise" or tokens[-4] == "music" or tokens[-4] == "babble" or tokens[-4] == "reverb":
            orig_utt_name = "-".join(tokens[:-4])
        else:
            orig_utt_name = "-".join(tokens[:-3])
        # orig_utt_name = orig_utt_name.split("%")[-1]
        # print (self.ali_dict)
        phn_label = self.ali_dict[orig_utt_name][begin_idx+7: begin_idx + utt_len -7]

        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean) / self.global_std
        return feat, name, [spk_id, phn_label]

    def __len__(self):
        return self.length

        
class KaldiDatasetUtt(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1 !
    """
    def __init__(self, feat_scp, stream, global_mean=None, global_std=None):
        super(KaldiDatasetUtt, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(stream.format(feat_scp))
        self.global_mean = global_mean
        self.global_std = global_std
        count = 0
        with open(feat_scp) as in_file:
            for line in in_file:
                count += 1
        self.length = count

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        spk_id = int(name.split("-")[-1])
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean) / self.global_std
        return feat, name, spk_id

    def __len__(self):
        return self.length

class KaldiDatasetUtt_ARK(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1 !
    """
    def __init__(self, feat_ark, stream, length=1000, global_mean=None, global_std=None):
        super(KaldiDatasetUtt_ARK, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(stream.format(feat_ark))
        self.global_mean = global_mean
        self.global_std = global_std
        count = 0
        self.length = length

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        spk_id = int(name.split("-")[-1])
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean) / self.global_std
        return feat, name, spk_id

    def __len__(self):
        return self.length

class KaldiDatasetUtt_Eval(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1 !
    """
    def __init__(self, feat_scp, stream, global_mean=None, global_std=None, min_len=25):
        super(KaldiDatasetUtt_Eval, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(stream.format(feat_scp))
        self.global_mean = global_mean
        self.global_std = global_std
        self.min_len = min_len

        count = 0
        with open(feat_scp) as in_file:
            for line in in_file:
                count += 1
        self.length = count
        print (self.length)

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean) / self.global_std
        if len(feat) < self.min_len:
            left_pad = ((self.min_len) - len(feat)) // 2
            right_pad = self.min_len - len(feat) - left_pad 
            feat = np.pad(feat, ((left_pad, right_pad),(0,0)), 'edge')

        return feat, name

    def __len__(self):
        return self.length

def read_label(label_file, label2int=None):
    """
    :param label_file: label files such as utt2spk
    :param label2int: For NN training, we need to transform the labels into int.
    :return:
    """
    labels_dict = {}
    with open(label_file, 'r') as read_label:
        for line in read_label:
            tokens = line.strip().split()
            labels_dict[tokens[0]] = tokens[1]
    if label2int is not None:
        labels_dict = {key: label2int[value] for key, value in labels_dict.items()}
    return labels_dict

def read_alignment(ali_ark_file, label2int=None):
    """
    :param label_file: label files such as utt2spk
    :param label2int: For NN training, we need to transform the labels into int.
    :return:
    """
    labels_dict = {key:vec for key, vec in kaldi_io.read_ali_ark(ali_ark_file)}
    if label2int is not None:
        for key, vec in labels_dict.items():
            vec = np.array([label2int[str(value)] for value in vec])
            labels_dict[key] = vec
    return labels_dict

def encode_labels(unique_label_file):
    """
    Encode string labels into int counting from 0.
    """
    with open(unique_label_file) as readLabels:
            labels = readLabels.read().splitlines()
            label_dict = {labels[idx]: idx for idx in range(len(labels))}
    return label_dict
