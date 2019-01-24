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
import torch.nn.init as init


def weight_init(m, conv_init=init.kaiming_normal_, convt_init=init.kaiming_normal_,
                bias_init=init.normal_, fc_init=init.xavier_normal_, rnn_init=init.orthogonal_):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        conv_init(m.weight.data)
        if m.bias is not None:
            bias_init(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        conv_init(m.weight.data)
        if m.bias is not None:
            bias_init(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        conv_init(m.weight.data)
        if m.bias is not None:
            bias_init(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        convt_init(m.weight.data)
        if m.bias is not None:
            bias_init(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        convt_init(m.weight.data)
        if m.bias is not None:
            bias_init(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        convt_init(m.weight.data)
        if m.bias is not None:
            bias_init(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        fc_init(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                rnn_init(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                rnn_init(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                rnn_init(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                rnn_init(param.data)
            else:
                init.normal_(param.data)