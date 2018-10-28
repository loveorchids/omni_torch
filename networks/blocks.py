import torch
import torch.nn.functional as tf
import torch.nn as tn
import networks.misc as misc

class ConvBlock(tn.Module):
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
                   activation = tn.ReLU, batch_norm = True, end_op=None):
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
        

    def conv_block(self):
        block = []
        # When an (HxWxC) image converted to torch tensor, the shape will become (CxHxW)
        # That is why we get the shape[1]
        repeat = self.repeat
        filters = [input.shape[1]] + self.filters * repeat
        kernel_sizes = self.kernel_size * repeat
        stride = self.stride * repeat
        padding = self.padding * repeat
        groups = self.group * repeat
        for i in range(len(filters)-1):
            if stride[i] >= 1:
                block.append(tn.Conv2d(in_channels=filters[i], out_channels=filters[i+1],
                                 kernel_size=kernel_sizes[i], stride=stride[i],
                                 padding=padding[i], groups=groups[i]))
            else:
                block.append(tn.ConvTranspose2d(in_channels=filters[i], out_channels=filters[i+1],
                                          kernel_size=kernel_sizes[i], stride=round(1/stride[i]),
                                          padding=padding[i], groups=groups[i]))
            if self.activation:
                block.append(self.activation())
            if self.batch_norm:
                block.append(tn.BatchNorm2d(filters[i+1]))
        return tn.Sequential(*block)
        #return model(input)
    
    def forward(self, input):
        out = self.conv(input)
        return out
    
class ResBlock(ConvBlock):
    def __init__(self, input, filters, kernel_sizes, stride, padding, repeat, groups,
                   activation = tn.ReLU, batch_norm = True, end_op=None):
        super(ResBlock, self).__init__()
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
        
    
    def resnet_block(self):
        conv = self.conv_block()
        if input.shape is not conv.shape:
            #a = torch.zeros([8, conv.shape[1]-input.shape[1], 32, 32], dtype=torch.float32)
            #shortcut = torch.cat([input, a], dim=1)
            # P stands for padding
            # S stands for stride
            S, P, K = misc.get_stride_padding_kernel(input.shape[2], conv.shape[2])
            shortcut = tn.Sequential(tn.Conv2d(in_channels=input.shape[1],
                                               out_channels=conv.shape[1],
                                               kernel_size=K, stride=S, padding=P),
                                     tn.BatchNorm2d(conv.shape[1]))
            self.param.append(shortcut.parameters())
            conv += shortcut(input)
        else:
            conv += input
        return tf.relu(conv)
    
    def forward(self, input):
        out =
    
    def pooling(self, input, mode=tn.MaxPool2d, kernel=2, stride=2, padding=0, dilation=1):
        pool = mode(kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)
        self.param.append(pool.parameters())
        return pool(input)
    
    def fc_layer(self, input, layer_size, activation = tn.ReLU, batch_norm = True):
        net = input.view(input.size(0), -1)
        layers = []
        layer_size.insert(0, net.size(1))
        for i in range(len(layer_size) - 1):
            layers.append(tn.Linear(layer_size[i], layer_size[i+1]))
            if activation:
                layers.append(activation())
            if batch_norm:
                layers.append(tn.BatchNorm2d(layer_size[i + 1]))
        model = tn.Sequential(*layers)
        self.param.append(model.parameters())
        return model(net)
    