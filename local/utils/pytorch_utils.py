#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Created Time:   2019-06-03
# @Author: Shuai Wang
# @Email: wsstriving@gmail.com
# @Last Modified Time: 2019-06-05

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        # print (target.unsqueeze(1))
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.sum(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingV2(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingV2, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None
    def forward(self, x, target):
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (x.size(1) - 1 ))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class MultiHotCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def test():
    lsm = LabelSmoothingLoss(0.05)
    data = torch.rand(3, 5)
    target = torch.LongTensor([0,1,2])
    print (lsm(data, target))

# test()