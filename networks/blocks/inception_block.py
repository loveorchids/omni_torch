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

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_block import conv_block

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

