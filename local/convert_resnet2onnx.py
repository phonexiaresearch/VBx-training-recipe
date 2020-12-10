#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import onnxruntime

from models.resnet2 import ResNet34, ResNet18, ResNet101


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to torch net.')
    parser.add_argument('-o', '--output', required=True, help='Output path.')
    parser.add_argument('--use-whole-net', required=False, action='store_true', default=False,
        help='use whole net together with metric')
    parser.add_argument('--metric', choices=['add_margin', 'arc_margin', 'sphere', 'linear'], required=False, default='linear')
    args = parser.parse_args()

    embed_dim = 256
    feat_dim = 64
    num_targets = 1000

    model = ResNet101(feat_dim, embed_dim)
    metric_fc = AddMarginProduct(embed_dim, num_targets, s=32, m=0.2)
    model.add_module('metric', metric_fc)
    state_dict = torch.load(args.input, map_location='cpu')['state_dict']
    del state_dict['metric.weight']
    
    if not args.use_whole_net:
        if 'metric.weight' in state_dict.keys():
            del state_dict['metric.weight']
        if 'metric.0.weight' in state_dict.keys():
            del state_dict['metric.0.weight']
        if 'metric.0.bias' in state_dict.keys():
            del state_dict['metric.0.bias']
        if 'metric.0.running_mean' in state_dict.keys():
            del state_dict['metric.0.running_mean']
        if 'metric.0.running_var' in state_dict.keys():
            del state_dict['metric.0.running_var']
        if 'metric.0.num_batches_tracked' in state_dict.keys():
            del state_dict['metric.0.num_batches_tracked']
        if 'metric.2.weight' in state_dict.keys():
            del state_dict['metric.2.weight']
        if 'metric.2.bias' in state_dict.keys():
            del state_dict['metric.2.bias']
    else:
        if args.metric == 'add_margin':
            metric_fc = AddMarginProduct(args.embed_dim, args.num_targets, s=32, m=0.2)
        elif args.metric == 'arc_margin':
            metric_fc = ArcMarginProduct(args.embed_dim, args.num_targets, s=32, m=0.2)
        elif args.metric == 'sphere':
            metric_fc = SphereProduct(args.embed_dim, args.num_targets, m=4)
        elif args.metric == 'linear':
            metric_fc = nn.Sequential(nn.BatchNorm1d(embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(embed_dim, 2))
        model.add_module('metric', metric_fc)

    model.load_state_dict(state_dict, strict=False)
    if args.use_whole_net:
        model = torch.nn.Sequential(model, model.metric)
    print(model)
    
    model.eval()

    x_numpy = np.random.rand(feat_dim, 1000)
    x_numpy2 = np.random.rand(feat_dim, 100)
    #x1 = torch.tensor(x_numpy).float() 
    #x2 = torch.tensor(x_numpy).float() 
    x1 = torch.tensor(x_numpy[np.newaxis, :, :]).float()
    x2 = torch.tensor(x_numpy[np.newaxis, :, :]).float()
    #x = torch.randn(batch_size, 40, 100, requires_grad=False)
    
    pytorch_embed = model.forward(x1)
    if not args.use_whole_net:
        pytorch_embed = pytorch_embed.squeeze()

    net_path = args.output
    torch.onnx.export(model, x2, net_path, input_names=['x'], dynamic_axes={'x': [2]})
    sess = onnxruntime.InferenceSession(net_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    onnx_xvec = sess.run([label_name], {input_name: x_numpy2[np.newaxis, :, :].astype(np.float32)})[0].squeeze()
    print(onnx_xvec -pytorch_embed.detach().numpy())
    print(onnx_xvec.shape)
    #print(np.allclose(onnx_xvec, pytorch_embed.detach().numpy()))



