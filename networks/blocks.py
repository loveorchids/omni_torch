import torch.nn.functional as tf
import torch.nn as tn



def conv_block(input, filters, kernel_sizes, stride, padding, repeat, groups,
               activation = tn.ReLU, batch_norm = True):
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
    assert max([len(filters), len(kernel_sizes), len(stride), len(padding), len(groups)]) \
           is \
           min([len(filters), len(kernel_sizes), len(stride), len(padding), len(groups)])
    assert len(filters) is repeat
    block = []
    # When an (HxWxC) image converted to torch tensor, the shape will become (CxHxW)
    # That is why we get the shape[1]
    filters = list(filters).insert(0, input.shape[1])
    for _ in range(repeat):
        for i in range(len(filters)-1):
            if stride[i] >= 1:
                block.append(tn.Conv2d(in_channels=filters[i], out_channels=filters[i+1],
                                 kernel_size=kernel_sizes[i], stride=stride[i],
                                 padding=padding[i], groups=groups[i]))
            else:
                block.append(tn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i+1],
                                          kernel_size=kernel_sizes[i], stride=round(1/stride[i]),
                                          padding=padding[i], groups=groups[i]))
            activation()
            if batch_norm:
                block.append(tn.BatchNorm2d(filters[i+1]))
    return tn.Sequential(*block)


def resnet_block(input, filters, kernel_sizes, stride, padding, repeat, groups,
               activation = tn.ReLU, batch_norm = True):
    assert max([len(filters), len(kernel_sizes), len(stride), len(padding), len(groups)]) \
           is \
           min([len(filters), len(kernel_sizes), len(stride), len(padding), len(groups)])
    conv = conv_block(input, filters, kernel_sizes, stride, padding, repeat, groups,
               activation, batch_norm)
    out = tf.relu(input)
    out = tn.BatchNorm2d(filters[-1])(out)
    out += conv(input)
    return tf.relu(out)