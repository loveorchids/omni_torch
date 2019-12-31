"""
# Copyright (c) 2019 Wang Hanqin
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

import torch.nn as nn
import torch.nn.functional as F
from .conv_block import conv_block
from .inception_block import InceptionBlock

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