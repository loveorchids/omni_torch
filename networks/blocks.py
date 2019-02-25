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
import torch.nn.functional as F

class Resnet_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, groups=1, name='',
                 dilation=1, bias=True, activation=nn.ReLU(), batch_norm=nn.BatchNorm2d,
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
                                     dropout=dropout, dilation=dilation, bias=bias)
        if shortcut_stride < 1:
            kernel_size, padding = 2, 0
        else:
            kernel_size, padding = 1, 0
        #TODO: dimesion of input param
        self.shortcut = conv_block(input, [filters[-1]], kernel_sizes=kernel_size, stride=shortcut_stride,
                                   padding=padding, groups=[groups[-1]], name="shortcut",
                                   activation=activation, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))


class InceptionBlock(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, groups=1, name=None,
                 dilation=1, bias=True, activation=nn.ReLU(), batch_norm=nn.BatchNorm2d,
                 dropout=0, inner_maxout= None, maxout=None):
        def concatenate_blocks(block1, block2):
            list_of_block1 = list(block1.children())
            list_of_block1.extend(list(block2.children()))
            return nn.Sequential(*list_of_block1)
        """
        :param input: int
        :param filters: in the form of [[...], [...], ... , [...]], each cell represent a stream in the network
        :param kernel_sizes: in the form of [[...], [...], ... , [...]]
        :param stride: in the form of [[...], [...], ... , [...]]
        :param padding: in the form of [[...], [...], ... , [...]]
        """
        assert max([len(filters), len(kernel_sizes), len(stride), len(padding)]) is \
               min([len(filters), len(kernel_sizes), len(stride), len(padding)])
        inner_groups = len(filters)
        super().__init__()
        if inner_maxout is None:
            inner_maxout = inner_groups * [None]
        inner_blocks = []
        for i in range(inner_groups):
            if inner_maxout[i]:
                ops = nn.Sequential(inner_maxout[i])
                ops = concatenate_blocks(ops, conv_block(input, filters[i], kernel_sizes[i], stride[i], padding[i],
                                                         name="incep_" + str(i), activation=activation,
                                                         batch_norm=batch_norm, dropout=dropout, dilation=dilation,
                                                         bias=bias, groups=groups))
            else:
                ops = conv_block(input, filters[i], kernel_sizes[i], stride[i], padding[i],
                                 name="incep_" + str(i), activation=activation, batch_norm=batch_norm,
                                 dropout=dropout, dilation=dilation, bias=bias, groups=groups)
            inner_blocks.append(ops)
        if maxout:
            inner_blocks.append(maxout)
        self.inner_blocks = nn.ModuleList(inner_blocks)

    def forward(self, x):
        out = [block(x) for block in self.inner_blocks]
        return torch.cat(out, dim=1)


class Xception_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, name=None, activation=nn.ReLU(),
                 batch_norm=None, dilation=1, bias=True, dropout=0, inner_maxout=None, maxout=None):
        super().__init__()
        self.conv_block = InceptionBlock(input, filters=filters, kernel_sizes=kernel_sizes, stride=stride,
                                         padding=padding, name=name, activation=activation, batch_norm=batch_norm,
                                         dilation=dilation, bias=bias, inner_maxout=inner_maxout, maxout=maxout)
        self.shortcut = resnet_shortcut(input, filters[-1], dropout=dropout)

    def forward(self, x):
        return F.relu(self.dropout(self.conv_block(x)) + self.shortcut(x))


class Conv_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, groups=1, name='',
               dilation=1, bias=True, activation=nn.ReLU(), batch_norm=None, dropout=0):
        super().__init__()
        self.layers = conv_block(input, filters, kernel_sizes, stride, padding, groups, name, dilation,
               bias, activation, batch_norm, dropout)

    def forward(self, x):
        return self.layers.forward(x)
    
class FC_Layer(nn.Module):
    def __init__(self, input, layer_size, bias=True, name=None, activation=nn.Sigmoid(),
                 batch_norm=None, dropout=0):
        super().__init__()
        self.fc_layer = fc_layer(input, layer_size, bias=bias, name=name, activation=activation,
                 batch_norm=batch_norm, dropout=dropout)
        
    def forward(self, x, batch_dim=0):
        if len(x.shape):
            x = x.view(x.size(batch_dim), -1)
        return self.fc_layer.forward(x)

def conv_block(input, filters, kernel_sizes, stride, padding, groups=1, name='', dilation=1,
               bias=True, activation=nn.ReLU(), batch_norm=None, dropout=0):
    """
    Create a convolutional block with several layers
    :param input: input data channels
    :param filters: int or list
    :param kernel_sizes: int or list
    :param stride: int or list
    :param padding: int or list
    :param groups: int or list, default 1
    :param name:
    :param activation:
    :param batch_norm:
    :return: nn.Sequential Object
    """
    filters = [input] + [filters] if type(filters) is not list else [input] + filters
    assert_length = len(filters) - 1
    kernel_sizes = standardize(kernel_sizes, assert_length)
    stride = standardize(stride, assert_length)
    padding = standardize(padding, assert_length)
    groups = standardize(groups, assert_length)
    dilation = standardize(dilation, assert_length)
    bias = standardize(bias, assert_length)
    activation = standardize(activation, assert_length)
    batch_norm = standardize(batch_norm, assert_length)
    dropout = standardize(dropout, assert_length)

    modules = nn.Sequential()
    for i in range(len(filters) - 1):
        if stride[i] >= 1:
            modules.add_module(name + "conv_" + str(i),
                               nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1],
                                         kernel_size=kernel_sizes[i], stride=stride[i], padding=padding[i],
                                         dilation=dilation[i], groups=groups[i], bias=bias[i]))
        else:
            modules.add_module(name + "conv_" + str(i),
                               nn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i + 1],
                                                  kernel_size=kernel_sizes[i], stride=round(1 / stride[i]),
                                                  padding=padding[i], dilation=dilation[i], groups=groups[i],
                                                  bias=bias[i]))
        if batch_norm[i]:
            modules.add_module(name + "bn_" + str(i), batch_norm[i](filters[i + 1]))
        if activation[i]:
            modules.add_module(name + "act_" + str(i), activation[i])
        if dropout[i] > 0:
            modules.add_module(name + "drop_" + str(i), nn.Dropout2d(dropout[i]))
    return modules

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

def fc_layer(input, layer_size, bias=True, name=None, activation=nn.Sigmoid(),
             batch_norm=None, dropout=0):
    layer_size = [input] + [layer_size] if type(layer_size) is not list else [input] + layer_size
    assert_length = len(layer_size) - 1
    bias = standardize(bias, assert_length)
    activation = standardize(activation, assert_length)
    batch_norm = standardize(batch_norm, assert_length)
    dropout = standardize(dropout, assert_length)
    
    if name is None:
        name = ""
    modules = nn.Sequential()
    layer_size = [input] + layer_size
    for i in range(len(layer_size) - 1):
        modules.add_module(name + "_fc_" + str(i), nn.Linear(layer_size[i], layer_size[i + 1], bias[i]))
        if batch_norm[i]:
            modules.add_module(name + "bn_" + str(i), batch_norm[i](layer_size[i + 1]))
        if activation[i]:
            modules.add_module(name + "act_" + str(i), activation[i])
        if dropout[i] > 0:
            modules.add_module(name + "drop_" + str(i), nn.Dropout2d(dropout[i]))
    return modules

def standardize(param, assert_length):
    if type(param) is not list and type(param) is not tuple:
        param = [param] * assert_length
    assert len(param) == assert_length, "expect %s input params, got %s input parameter" \
                                        % (assert_length, len(param))
    return param