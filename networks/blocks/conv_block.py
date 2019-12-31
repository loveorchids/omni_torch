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
from .utils import standardize

class Conv_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, groups=1, name='', dilation=1,
                 bias=True, activation=nn.ReLU(), batch_norm=None, dropout=0, transpose=False):
        super().__init__()
        self.layers = conv_block(input, filters, kernel_sizes, stride, padding, groups, name, dilation,
               bias, activation, batch_norm, dropout, transpose)

    def forward(self, x):
        return self.layers.forward(x)
    

def conv_block(input, filters, kernel_sizes, stride, padding, groups=1, name='', dilation=1,
               bias=True, activation=nn.ReLU(), batch_norm=None, dropout=0, transpose=False):
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
    transpose = standardize(transpose, assert_length)

    modules = nn.Sequential()
    for i in range(len(filters) - 1):
        if transpose[i]:
            modules.add_module("convT_" + name + "_" + str(i),
                               nn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i + 1],
                                                  kernel_size=kernel_sizes[i], stride=stride[i], padding=padding[i],
                                                  dilation=dilation[i], groups=groups[i], bias=bias[i]))
        else:
            modules.add_module("conv_" + name + "_" + str(i),
                               nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1],
                                         kernel_size=kernel_sizes[i], stride=stride[i], padding=padding[i],
                                         dilation=dilation[i], groups=groups[i], bias=bias[i]))

        if batch_norm[i]:
            modules.add_module("bn_" + name + "_" + str(i), batch_norm[i](filters[i + 1]))
        if activation[i]:
            modules.add_module("act_" + name + "_" + str(i), activation[i])
        if dropout[i] > 0:
            modules.add_module("drop_" + name + "_" + str(i), nn.Dropout2d(dropout[i]))
    return modules