"""
# Copyright (c) 2018 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import torch
import torch.nn as nn
import torch.nn.functional as tf

class Resnet_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, groups=None,
                 name=None, activation=nn.ReLU(), batch_norm=True, bn_eps=1e-5,
                 dropout=0, shortcut_stride=1):
        """
        :param input: int
        :param filters: in the form of [[...], [...], ... , [...]]
        :param kernel_sizes: in the form of [[...], [...], ... , [...]]
        :param stride: in the form of [[...], [...], ... , [...]]
        :param padding: in the form of [[...], [...], ... , [...]]
        """
        super().__init__()
        self.conv_block = conv_block(input, filters, kernel_sizes, stride, padding, groups=groups,
                                     name=name, activation=activation, batch_norm=batch_norm,
                                     bn_eps=bn_eps, dropout=dropout)
        if shortcut_stride < 1:
            kernel_size, padding = 2, 0
        else:
            kernel_size, padding = 1, 0
        self.shortcut = conv_block(input, [filters[-1]], kernel_sizes=[kernel_size], stride=[shortcut_stride],
                                   padding=[padding], groups=[groups[-1]], name=name+"_shortcut",
                                   activation=activation, batch_norm=batch_norm, bn_eps=bn_eps, dropout=dropout)

    def forward(self, x):
        return tf.relu(self.conv_block(x) + self.shortcut(x))


class InceptionBlock(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding,
                 name=None, activation=nn.ReLU(), batch_norm=True, bn_eps=1e-5,
                 dropout=0, inner_maxout= None, maxout=None):
        """
        :param input: int
        :param filters: in the form of [[...], [...], ... , [...]]
        :param kernel_sizes: in the form of [[...], [...], ... , [...]]
        :param stride: in the form of [[...], [...], ... , [...]]
        :param padding: in the form of [[...], [...], ... , [...]]
        """
        assert max([len(filters), len(kernel_sizes), len(stride), len(padding)]) is \
               min([len(filters), len(kernel_sizes), len(stride), len(padding)])
        inner_groups = len(filters)
        super().__init__()
        if name is None:
            name = ""
        if inner_maxout is None:
            inner_maxout = inner_groups * [None]
        inner_blocks = []
        for i in range(inner_groups):
            if inner_maxout[i]:
                ops = nn.Sequential(inner_maxout[i])
                ops = concatenate_blocks(ops, conv_block(input, filters[i], kernel_sizes[i], stride[i], padding[i],
                                                         name=name + "_inner_" + str(i), activation=activation,
                                                         batch_norm=batch_norm, bn_eps=bn_eps, dropout=dropout))
            else:
                ops = conv_block(input, filters[i], kernel_sizes[i], stride[i], padding[i],
                                 name=name + "_inner_" + str(i), activation=activation,
                                 batch_norm=batch_norm, bn_eps=bn_eps, dropout=dropout)
            inner_blocks.append(ops)
        if maxout:
            inner_blocks.append(maxout)
        self.inner_blocks = nn.ModuleList(inner_blocks)

    def forward(self, x):
        out = [block(x) for block in self.inner_blocks]
        return torch.cat(out, dim=1)


class Xception_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding,
                 name=None, activation=  nn.ReLU(), batch_norm=True, bn_eps=1e-5,
                 dropout=0, inner_maxout=None, maxout=None):
        super().__init__()
        self.conv_block = InceptionBlock(input, filters=filters, kernel_sizes=kernel_sizes,
                                         stride=stride, padding=padding, name=name,
                                         activation=activation, batch_norm=batch_norm,
                                         inner_maxout=inner_maxout, maxout=maxout, bn_eps=bn_eps)
        self.shortcut = resnet_shortcut(input, filters[-1], dropout=dropout)

    def forward(self, x):
        return tf.relu(self.dropout(self.conv_block(x)) + self.shortcut(x))


def conv_block(input, filters, kernel_sizes, stride, padding, groups=None, repeat=1,
               name=None, activation=nn.ReLU(), batch_norm=True, bn_eps=1e-5, dropout=0):
    """
    Create a convolutional block with several layers
    :param input: input data channels
    :param filters: int or list
    :param kernel_sizes: int or list
    :param stride: int or list
    :param padding: int or list
    :param groups: int or list, default None. If None, then
    :param repeat: int, default 1, if several block with same architecture need to be created
    :param name:
    :param activation:
    :param batch_norm:
    :param bn_eps:
    :return:
    """
    filters = [filters] if type(filters) is not list else filters
    kernel_sizes = [kernel_sizes] if type(kernel_sizes) is not list else kernel_sizes
    stride = [stride] if type(stride) is not list else stride
    padding = [padding] if type(padding) is not list else padding
    assert max([len(filters), len(kernel_sizes), len(stride), len(padding)]) is \
           min([len(filters), len(kernel_sizes), len(stride), len(padding)])
    if groups is None:
        groups = [1] * len(filters)
    else:
        groups = [groups] if type(groups) is not list else groups
    if type(dropout) is list or type(dropout) is tuple:
        assert len(dropout)==len(filters)
        dropout = dropout * repeat
    if type(activation) is list or type(activation) is tuple:
        assert len(activation)==len(filters)
        activation = activation * repeat
    ops = nn.Sequential()

    filters = [input] + filters * repeat
    kernel_sizes = kernel_sizes * repeat
    stride = stride * repeat
    padding = padding * repeat
    groups = groups * repeat

    if name is None:
        name = ""
    for i in range(len(filters) - 1):
        if stride[i] >= 1:
            ops.add_module(name + "_conv_" + str(i),
                           nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1],
                                     kernel_size=kernel_sizes[i], stride=stride[i],
                                     padding=padding[i], groups=groups[i]))
        else:
            ops.add_module(name + "_convT_" + str(i),
                           nn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i + 1],
                                              kernel_size=kernel_sizes[i], stride=round(1 / stride[i]),
                                              padding=padding[i], groups=groups[i]))
        if type(dropout) is int:
            if dropout > 0:
                ops.add_module(name + "_dropout_"+ str(i), nn.Dropout2d(dropout))
        if type(dropout) is list or type(dropout) is tuple:
            if dropout[i] > 0:
                ops.add_module(name + "_dropout_"+ str(i), nn.Dropout2d(dropout[i]))
        if batch_norm:
            if type(batch_norm) is str and batch_norm.lower() == "instance":
                ops.add_module(name + "_InsNorm_" + str(i), nn.InstanceNorm2d(filters[i + 1], eps=bn_eps))
            else:
                ops.add_module(name + "_BthNorm_" + str(i), nn.BatchNorm2d(filters[i + 1], eps=bn_eps))
        if type(activation) is list or type(activation) is tuple:
            if activation[i]:
                ops.add_module(name + "_active_" + str(i), activation[i])
        else:
            if activation:
                ops.add_module(name + "_active_" + str(i), activation)
    return ops

def resnet_shortcut(input, output, kernel_size=1, stride=1, padding=0,
                    batch_norm=True, bn_eps=1e-5, dropout=0, name=None):
    if name is None:
        name = ""
    ops = nn.Sequential()
    # S, P, K = misc.get_stride_padding_kernel(input.shape[2], conv.shape[2])
    ops.add_module(name + "_shortcut_conv",
                   nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size,
                             stride=stride, padding=padding))
    if dropout > 0:
        ops.add_module(name + "_dropout", nn.Dropout2d(dropout))
    if batch_norm:
        if type(batch_norm) is str and batch_norm.lower() == "instance":
            ops.add_module(name + "_shortcut_InsNorm", nn.InstanceNorm2d(output, eps=bn_eps))
        else:
            ops.add_module(name + "_shortcut_BthNorm", nn.BatchNorm2d(output, eps=bn_eps))
    return ops

def fc_layer(input, layer_size, name=None, activation=nn.Sigmoid(), batch_norm=True, bn_eps=1e-5,
             dropout=0):
    if name is None:
        name = ""
    ops = nn.Sequential()
    layer_size = [input] + layer_size
    for i in range(len(layer_size) - 1):
        ops.add_module(name + "_fc_" + str(i), nn.Linear(layer_size[i], layer_size[i + 1]))
        if type(dropout) is int:
            if dropout > 0:
                ops.add_module(name + "_dropout_"+ str(i), nn.Dropout2d(dropout))
        if type(dropout) is list or type(dropout) is tuple:
            if dropout[i] > 0:
                ops.add_module(name + "_dropout_"+ str(i), nn.Dropout2d(dropout[i]))
        if batch_norm:
            if type(batch_norm) is str and batch_norm.lower() == "instance":
                ops.add_module(name + "_BN_" + str(i), nn.InstanceNorm1d(layer_size[i + 1], eps=bn_eps))
            else:
                ops.add_module(name + "_BN_" + str(i), nn.BatchNorm1d(layer_size[i + 1], eps=bn_eps))
        if type(activation) is list or type(activation) is tuple:
            if activation[i]:
                ops.add_module(name + "_active_" + str(i), activation[i])
        else:
            if activation:
                ops.add_module(name + "_active_" + str(i), activation)
    return ops

def concatenate_blocks(block1, block2):
    list_of_block1 = list(block1.children())
    list_of_block1.extend(list(block2.children()))
    return nn.Sequential(*list_of_block1)