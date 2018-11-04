import torch
import torch.nn.functional as tf
import torch.nn as nn
import networks.misc as misc


def conv_block(input, filters, repeat, kernel_sizes=(3, 3), stride=(1, 1),
               padding=(1, 1), groups=(1, 1), activation = nn.ReLU, batch_norm = True,
               end_op=None):
    ops = []
    # When an (HxWxC) image converted to torch tensor, the shape will become (CxHxW)
    # That is why we get the shape[1]
    filters = [input] + filters * repeat
    kernel_sizes = kernel_sizes * repeat
    stride = stride * repeat
    padding = padding * repeat
    groups = groups * repeat
    for i in range(len(filters)-1):
        if stride[i] >= 1:
            ops.append(nn.Conv2d(in_channels=filters[i], out_channels=filters[i+1],
                             kernel_size=kernel_sizes[i], stride=stride[i],
                             padding=padding[i], groups=groups[i]))
        else:
            ops.append(nn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i+1],
                                      kernel_size=kernel_sizes[i], stride=round(1/stride[i]),
                                      padding=padding[i], groups=groups[i]))
        if activation:
            ops.append(activation())
        if batch_norm:
            ops.append(nn.BatchNorm2d(filters[i+1]))
    return nn.Sequential(*ops)
    
def resnet_shortcut(input, output, kernel_size=1, stride=1, padding=0, batch_norm=True):
    ops = []
    #S, P, K = misc.get_stride_padding_kernel(input.shape[2], conv.shape[2])
    ops.append(nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size,
                         stride=stride, padding=padding))
    if batch_norm:
        ops.append(nn.BatchNorm2d(output))
    return nn.Sequential(*ops)


def fc_layer(input, layer_size, activation = nn.ReLU, batch_norm = True):
    ops = []
    layer_size = [input] + layer_size
    for i in range(len(layer_size) - 1):
        ops.append(nn.Linear(layer_size[i], layer_size[i+1]))
        if activation:
            ops.append(activation())
        if batch_norm:
            ops.append(nn.BatchNorm2d(layer_size[i + 1]))
    return nn.Sequential(*ops)