import torch
import torch.nn as nn
from omni_torch.networks import blocks as omth_blocks

# See the vanilla implementation here:
# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

class InceptionNet_V3(nn.Module):
    def __init__(self, BN):
        super().__init__()
        pool_3_1_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_layers = omth_blocks.conv_block(3, filters=[32, 32, 64, 80, 192], repeat=1,
                                                  kernel_sizes=[3, 3, 3, 1, 3], stride=[2, 1, 1, 1, 1], padding=[0, 0, 1, 0, 0],
                                                  groups=[1]*5, name="conv_layers", batch_norm=BN)
        self.inception_A = omth_blocks.InceptionBlock(192, filters=[64, [48, 64], [64, 96, 96], 32], kernel_sizes=[1, [1, 5], [1, 3, 3], 1],
                                                      stride=[1, [1, 1], [1, 1, 1], 1], padding=[0, [0, 2], [0, 1, 1], 0], inner_groups=4, batch_norm=BN,
                                                      maxout=[None, None, None, pool_3_1_1])
