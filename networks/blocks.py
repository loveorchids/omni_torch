import torch
import torch.nn as nn
import torch.nn.functional as tf

class Resnet_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, groups,
               dropout=0.2, name=None, activation = nn.ReLU, batch_norm = True):
        super().__init__()
        # repeat always equals to 1 here, because we only create one Resnet Block
        self.conv_block = conv_block(input, filters, 1, kernel_sizes, stride, padding, groups,
               name, activation, batch_norm)
        self.shortcut = resnet_shortcut(input, filters[-1])
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return tf.relu(self.dropout(self.conv_block(x))+self.shortcut(x))

class InceptionBlock(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, inner_groups,
               dropout=0.2, name=None, activation = nn.ReLU, batch_norm = True):
        """
        :param input: int
        :param filters: in the form of [[...], [...], ... , [...]]
        :param kernel_sizes: in the form of [[...], [...], ... , [...]]
        :param stride: in the form of [[...], [...], ... , [...]]
        :param padding: in the form of [[...], [...], ... , [...]]
        :param inner_groups: int, number of inner blocks
        :param name:
        :param activation:
        :param batch_norm:
        """
        assert max([len(filters), len(kernel_sizes), len(stride), len(padding), inner_groups]) is \
               min([len(filters), len(kernel_sizes), len(stride), len(padding), inner_groups])
        super().__init__()
        self.inner_blocks = []
        for i in range(inner_groups):
            assert max([len(filters[i]), len(kernel_sizes[i]), len(stride[i]), len(padding[i])]) is \
                   min([len(filters[i]), len(kernel_sizes[i]), len(stride[i]), len(padding[i])])
            # repeat always equals to 1 here, because we only create one Inception Block
            self.inner_blocks.append(conv_block(input, filters[i], 1, kernel_sizes[i], stride[i],
                                                padding[i], groups=[1] * len(filters[i]), name=name+"_inner_" + str(i),
                                                activation=activation, batch_norm=batch_norm))
            self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        out = [self.dropour(block(x)) for block in self.inner_blocks]
        return torch.cat(out, dim=1)
    
class Xception_Block(nn.Module):
    def __init__(self, input, filters, kernel_sizes, stride, padding, inner_groups,
               dropout=0.2, name=None, activation = nn.ReLU, batch_norm = True):
        super().__init__()
        self.conv_block = InceptionBlock(input, filters, kernel_sizes, stride, padding,
                                         inner_groups, dropout, name, activation, batch_norm)
        self.shortcut = resnet_shortcut(input, filters[-1])
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return tf.relu(self.dropout(self.conv_block(x)) + self.shortcut(x))

def conv_block(input, filters, repeat, kernel_sizes, stride, padding, groups,
               name=None, activation = nn.ReLU, batch_norm = True):
    ops = nn.Sequential()
    # When an (HxWxC) image converted to torch tensor, the shape will become (CxHxW)
    # That is why we get the shape[1]
    filters = [input] + filters * repeat
    kernel_sizes = kernel_sizes * repeat
    stride = stride * repeat
    padding = padding * repeat
    groups = groups * repeat
    if name is None:
        name = ""
    for i in range(len(filters)-1):
        if stride[i] >= 1:
            ops.add_module(name + "_conv_" + str(i),
                           nn.Conv2d(in_channels=filters[i], out_channels=filters[i+1],
                                     kernel_size=kernel_sizes[i], stride=stride[i],
                                     padding=padding[i], groups=groups[i]))
        else:
            ops.add_module(name + "_convT_" + str(i),
                nn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i+1],
                                   kernel_size=kernel_sizes[i], stride=round(1/stride[i]),
                                   padding=padding[i], groups=groups[i]))
        if activation:
            ops.add_module(name + "_active_" + str(i), activation())
        if batch_norm:
            ops.add_module(name + "_BN_" + str(i), nn.BatchNorm2d(filters[i+1]))
    return ops

def nice_block(input, filters, repeat, kernel_sizes, stride, padding, group):
    conv_block(input, filters, repeat, kernel_sizes, stride, padding, group),
    pass
    
def resnet_shortcut(input, output, kernel_size=1, stride=1, padding=0,
                    batch_norm=True, name=None):
    if name is None:
        name = ""
    ops = nn.Sequential()
    #S, P, K = misc.get_stride_padding_kernel(input.shape[2], conv.shape[2])
    ops.add_module(name + "_shortcut_conv",
                   nn.Conv2d(in_channels=input, out_channels=output,
                             kernel_size=kernel_size,
                             stride=stride, padding=padding))
    if batch_norm:
        ops.add_module(name + "_shortcut_BN", nn.BatchNorm2d(output))
    return ops

def stn_block():
    pass

def fc_layer(input, layer_size, name=None, activation = nn.ReLU, batch_norm = True):
    if name is None:
        name = ""
    ops = nn.Sequential()
    layer_size = [input] + layer_size
    for i in range(len(layer_size) - 1):
        ops.add_module(name + "_fc_" + str(i),
                       nn.Linear(layer_size[i], layer_size[i+1]))
        if activation:
            ops.add_module(name + "_active_" + str(i), activation())
        if batch_norm:
            ops.add_module(name + "_BN_" + str(i),
                           nn.BatchNorm2d(layer_size[i + 1]))