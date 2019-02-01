import torch
import torch.nn as nn
from omni_torch.networks import blocks as omth_blocks

# See the vanilla implementation here:
# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

class InceptionNet_V3(nn.Module):
    def __init__(self, BN):
        super().__init__()
        avg_pool_3_1_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        pool_3_2_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv_layers = omth_blocks.conv_block(3, filters=[32, 32, 64, 80, 192], repeat=1,
                                                  kernel_sizes=[3, 3, 3, 1, 3], stride=[2, 1, 1, 1, 1], padding=[0, 0, 1, 0, 0],
                                                  groups=[1]*5, name="conv_layers", batch_norm=BN)
        
        self.Mixed5b = omth_blocks.InceptionBlock(192, filters=[64, [48, 64], [64, 96, 96], 32],
                                                  kernel_sizes=[1, [1, 5], [1, 3, 3], 1],  stride=[1, [1, 1], [1, 1, 1], 1],
                                                  padding=[0, [0, 2], [0, 1, 1], 0], inner_groups=4, batch_norm=BN,
                                                  inner_maxout=[None, None, None, avg_pool_3_1_1])
        self.Mixed5c = omth_blocks.InceptionBlock(256, filters=[64, [48, 64], [64, 96, 96], 64],
                                                  kernel_sizes=[1, [1, 5], [1, 3, 3], 1], stride=[1, [1, 1], [1, 1, 1], 1],
                                                  padding=[0, [0, 2], [0, 1, 1], 0], inner_groups=4, batch_norm=BN,
                                                  inner_maxout=[None, None, None, avg_pool_3_1_1])
        self.Mixed5d = omth_blocks.InceptionBlock(288, filters=[64, [48, 64], [64, 96, 96], 64],
                                                  kernel_sizes=[1, [1, 5], [1, 3, 3], 1], stride=[1, [1, 1], [1, 1, 1], 1],
                                                  padding=[0, [0, 2], [0, 1, 1], 0], inner_groups=4, batch_norm=BN,
                                                  inner_maxout=[None, None, None, avg_pool_3_1_1])
        
        
        self.Mixed6a = omth_blocks.InceptionBlock(288, filters=[384, [64, 96, 96]],
                                                  kernel_sizes=[3, [1, 3, 3]], stride=[2, [1, 1, 2]], padding=[0, [0, 1, 0]],
                                                  inner_groups=2, maxout=pool_3_2_0)
        self.Mixed6b = omth_blocks.InceptionBlock(768, filters=[192, [120, 120, 192], [120, 120, 120, 120, 192], 192],
                                                  kernel_sizes=[1, [1, [1, 7], [7, 1]], [1, [7, 1], [1, 7], [7, 1], [1, 7]], 1],
                                                  stride=[1, [1, 1, 1], [1, 1, 1 ,1], 1],
                                                  padding=[0, [0, [0, 3], [3, 0]], [0, [0, 3], [3, 0], [0, 3], [3, 0]], 0],
                                                  inner_groups=4, inner_maxout=[None, None, None, avg_pool_3_1_1])
        self.Mixed6c = omth_blocks.InceptionBlock(768, filters=[192, [160, 160, 192], [160, 160, 160, 160, 192], 192],
                                                  kernel_sizes=[1, [1, [1, 7], [7, 1]], [1, [7, 1], [1, 7], [7, 1], [1, 7]], 1],
                                                  stride=[1, [1, 1, 1], [1, 1, 1, 1], 1],
                                                  padding=[0, [0, [0, 3], [3, 0]], [0, [0, 3], [3, 0], [0, 3], [3, 0]], 0],
                                                  inner_groups=4, inner_maxout=[None, None, None, avg_pool_3_1_1])
        self.Mixed6d = omth_blocks.InceptionBlock(768, filters=[192, [160, 160, 192], [160, 160, 160, 160, 192], 192],
                                                  kernel_sizes=[1, [1, [1, 7], [7, 1]], [1, [7, 1], [1, 7], [7, 1], [1, 7]], 1],
                                                  stride=[1, [1, 1, 1], [1, 1, 1, 1], 1],
                                                  padding=[0, [0, [0, 3], [3, 0]], [0, [0, 3], [3, 0], [0, 3], [3, 0]], 0],
                                                  inner_groups=4, inner_maxout=[None, None, None, avg_pool_3_1_1])
        self.Mixed6e = omth_blocks.InceptionBlock(768, filters=[192, [192, 192, 192], [192, 192, 192, 192, 192], 192],
                                                  kernel_sizes=[1, [1, [1, 7], [7, 1]], [1, [7, 1], [1, 7], [7, 1], [1, 7]], 1],
                                                  stride=[1, [1, 1, 1], [1, 1, 1, 1], 1],
                                                  padding=[0, [0, [0, 3], [3, 0]], [0, [0, 3], [3, 0], [0, 3], [3, 0]], 0],
                                                  inner_groups=4, inner_maxout=[None, None, None, avg_pool_3_1_1])

        
        self.Mixed7a = omth_blocks.InceptionBlock(768, filters=[[192, 320], [192, 192, 192, 192]],
                                                  kernel_sizes=[[1, 3], [1, [7, 1], [1, 7], 3]], stride=[[1, 1], [1, 1, 1, 1]],
                                                  padding=[[0, 1], [0, [0, 3], [3, 0], 1]],
                                                  inner_groups=2, maxout=pool_3_2_0)
        self.Mixed7b = omth_blocks.InceptionBlock(1280, filters=[320, [384, 384, 384], [448, 384, 384, 384], 192],
                                                  kernel_sizes=[1, [1, [1, 3], [3, 1]], [1, 3, [1, 3], [3, 1]], 1],
                                                  stride=[1, [1, 1, 1], [1, 1, 1, 1], 1], padding=[0, [0, [0, 1], [1, 0]], [0, 1, [0, 1], [1, 0],0], 0],
                                                  inner_groups=4, inner_maxout=[None, None, None, avg_pool_3_1_1])
        self.Mixed7b = omth_blocks.InceptionBlock(2048)
        
