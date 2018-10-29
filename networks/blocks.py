import torch
import torch.nn.functional as tf
import torch.nn as nn
import networks.misc as misc

class ConvBlock(nn.Module):
    """
    :param input: torch tensor, the input of the current conv layer block
    :param filters: tuple or list, decide the number of filters in each conv layer
    :param kernel_sizes: tuple or list, decide the number of filters in each conv layer
    :param stride: tuple or list, decide the stride of each conv layer
    :param padding: tuple or list, decide the padding in each conv layer
    :param repeat: int, decide how much layer in this conv block
    :param groups: see https://pytorch.org/docs/stable/nn.html#conv2d
    :param activation: tuple or list, decide the activation function in each conv layer
    :param batch_norm: bool, do batch norm in each layers
    :return: nn.Sequential
    """
    def __init__(self, input, filters, kernel_sizes, stride, padding, repeat, groups,
                   activation = nn.ReLU, batch_norm = True, end_op=None):
        super(ConvBlock, self).__init__()
        assert max([len(filters), len(kernel_sizes), len(stride), len(padding), len(groups)]) \
               is \
               min([len(filters), len(kernel_sizes), len(stride), len(padding), len(groups)])
        assert len(filters) is repeat
        
        self.input = input
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.padding = padding
        self.repeat = repeat
        self.groups = groups
        self.activation = activation
        self.batch_norm = batch_norm
        self.end_op = end_op
        
        self.conv = self.conv_block()
        

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
    #return model(input)
    

        
    
def resnet_shortcut(input, output, kernel_size=1, stride=1, padding=0, batch_norm=True):
    ops = []
    #S, P, K = misc.get_stride_padding_kernel(input.shape[2], conv.shape[2])
    ops.append(nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size,
                         stride=stride, padding=padding))
    if batch_norm:
        ops.append(nn.BatchNorm2d(output))
    return nn.Sequential(*ops)
    
    def forward(self, input):
        pass


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