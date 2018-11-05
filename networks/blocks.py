import torch.nn as nn


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
    
def resnet_shortcut(input, output, kernel_size=1, stride=1, padding=0,
                    batch_norm=True, name=None):
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
    return ops