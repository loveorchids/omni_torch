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
    for i in range(len(layer_size) - 1):
        modules.add_module(name + "_fc_" + str(i), nn.Linear(layer_size[i], layer_size[i + 1], bias[i]))
        if batch_norm[i]:
            modules.add_module(name + "bn_" + str(i), batch_norm[i](layer_size[i + 1]))
        if activation[i]:
            modules.add_module(name + "act_" + str(i), activation[i])
        if dropout[i] > 0:
            modules.add_module(name + "drop_" + str(i), nn.Dropout2d(dropout[i]))
    return modules
