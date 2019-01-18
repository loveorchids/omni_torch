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