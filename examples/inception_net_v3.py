import torch
import torch.nn as nn
import torch.nn.functional as tf
import omni_torch.networks.initialization as init
from omni_torch.networks import blocks as omth_blocks

# Compare with the vanilla implementation here:
# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

class InceptionNet_V3(nn.Module):
    def __init__(self, BN, num_classes=1000, aux_logits=True):
        """
        :param BN:
        :param num_classes:
        :param aux_logits:
        """
        super().__init__()
        self.avg_pool_3_1_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_3_2_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.pool_8_1_0 = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)
        self.BN = BN
        self.aux_logits = aux_logits

        self.conv_layers_1 = omth_blocks.conv_block(3, filters=[32, 32, 64], kernel_sizes=[3, 3, 3], stride=[2, 1, 1],
                                                  padding=[0, 0, 1], name="conv_block_1", batch_norm=BN)
        self.conv_layers_2 = omth_blocks.conv_block(64, filters=[80, 192], kernel_sizes=[1, 3], stride=[1, 1],
                                                  padding=[0, 0], name="conv_block_2", batch_norm=BN)
        self.Mixed5b = self.inceptionA(in_channels=192, pool_features=32, pool=self.avg_pool_3_1_1)
        self.Mixed5c = self.inceptionA(in_channels=256, pool_features=64, pool=self.avg_pool_3_1_1)
        self.Mixed5d = self.inceptionA(in_channels=288, pool_features=64, pool=self.avg_pool_3_1_1)
        self.Mixed6a = omth_blocks.InceptionBlock(288, filters=[[384], [64, 96, 96]], kernel_sizes=[[3], [1, 3, 3]],
                                                  stride=[[2], [1, 1, 2]], padding=[[0], [0, 1, 0]],
                                                  batch_norm=BN,  maxout=self.pool_3_2_0)
        self.Mixed6b = self.inceptionC(in_channels=768, c7=120, pool=self.avg_pool_3_1_1)
        self.Mixed6c = self.inceptionC(in_channels=768, c7=160, pool=self.avg_pool_3_1_1)
        self.Mixed6d = self.inceptionC(in_channels=768, c7=160, pool=self.avg_pool_3_1_1)
        self.Mixed6e = self.inceptionC(in_channels=768, c7=192, pool=self.avg_pool_3_1_1)

        self.Mixed7a = omth_blocks.InceptionBlock(768, filters=[[192, 320], [192, 192, 192, 192]],
                                                  kernel_sizes=[[1, 3], [1, [7, 1], [1, 7], 3]],
                                                  stride=[[1, 2], [1, 1, 1, 2]],
                                                  padding=[[0, 0], [0, [0, 3], [3, 0], 0]],
                                                  maxout=self.pool_3_2_0, batch_norm=BN)
        self.Mixed7b = InceptionE(1280, BN)
        self.Mixed7c = InceptionE(2048, BN)
        self.fc = nn.Linear(2048, num_classes)

        if aux_logits:
            self.aux_conv = omth_blocks.conv_block(768, filters=[128, 768], kernel_sizes=[1, 5], stride=[1, 1],
                                                   padding=[0, 0], batch_norm=BN)
            self.aux_conv.stdev = 0.01
            self.aux_fc = nn.Linear(768, num_classes)
            self.aux_fc.stddev = 0.001

    def inceptionA(self, in_channels, pool_features, pool):
        return omth_blocks.InceptionBlock(in_channels, filters=[[64], [48, 64], [64, 96, 96], [pool_features]],
                                          kernel_sizes=[[1], [1, 5], [1, 3, 3], [1]],
                                          stride=[[1], [1, 1], [1, 1, 1], [1]],
                                          padding=[[0], [0, 2], [0, 1, 1], [0]], batch_norm=self.BN,
                                          inner_maxout=[None, None, None, pool])
    def inceptionC(self, in_channels, c7, pool):
        return omth_blocks.InceptionBlock(in_channels, filters=[[192], [c7, c7, 192], [c7, c7, c7, c7, 192], [192]],
                                          kernel_sizes=[[1], [1, [1, 7], [7, 1]], [1, [7, 1], [1, 7], [7, 1], [1, 7]], [1]],
                                          stride=[[1], [1, 1, 1], [1, 1, 1 ,1, 1], [1]],
                                          padding=[[0], [0, [0, 3], [3, 0]], [0, [0, 3], [3, 0], [0, 3], [3, 0]], [0]],
                                          batch_norm=self.BN, inner_maxout=[None, None, None, pool])

    def forward(self, x):
        x = self.pool_3_2_0(self.conv_layers_1(x))
        x = self.pool_3_2_0(self.conv_layers_2(x))

        x = self.Mixed5b(x)
        x = self.Mixed5c(x)
        x = self.Mixed5d(x)

        x = self.Mixed6a(x)
        x = self.Mixed6b(x)
        x = self.Mixed6c(x)
        x = self.Mixed6d(x)
        x = self.Mixed6e(x)

        if self.training and self.aux_logits:
            aux = self.aux_conv(x)
            aux = self.aux_fc(aux)

        x = self.Mixed7a(x)
        x = self.Mixed7b(x)
        x = self.Mixed7c(x)
        x = self.pool_8_1_0(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux
        else:
            return x

class InceptionE(nn.Module):
    def __init__(self, in_channels, BN, bn_eps=1e-5):
        """
        Due to this inception block contains sub inception blocks, we have to define it independently.
        """
        super(InceptionE, self).__init__()
        self.branch1x1 = omth_blocks.conv_block(in_channels, filters=[320], kernel_sizes=[1], stride=[1],
                                                padding=[0], batch_norm=BN)

        self.branch3x3_1 = omth_blocks.conv_block(in_channels, filters=[384], kernel_sizes=[1], stride=[1],
                                                  padding=[0], batch_norm=BN)
        self.branch3x3_2 = self.sub_inception_module(BN)

        self.branch3x3dbl_1 = omth_blocks.conv_block(in_channels, filters=[384, 384], kernel_sizes=[1, 3], stride=[1, 1],
                                                     padding=[0, 1], batch_norm=BN)
        self.branch3x3dbl_2 = self.sub_inception_module(BN)

        self.branch_pool = omth_blocks.conv_block(in_channels, filters=[192], kernel_sizes=[1], stride=[1],
                                                  padding=[0], batch_norm=BN)

    def sub_inception_module(self, BN, bn_eps=1e-5):
        return omth_blocks.InceptionBlock(384, filters=[384, 384], kernel_sizes=[[[1, 3]], [[3, 1]]],
                                          stride=[1, 1], padding=[[[0, 1]], [[1, 0]]], batch_norm=BN)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_2(self.branch3x3_1(x))
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))

        branch_pool = tf.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)



if __name__ == "__main__":
    x = torch.randn(1, 3, 299, 299)
    inception_net1 = InceptionNet_V3(BN=nn.BatchNorm2d, num_classes=10, aux_logits=False)
    #inception_net1.apply(init.weight_init)

    y1 = inception_net1(x)
    print(y1.shape)