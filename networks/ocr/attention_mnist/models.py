import torch
import torch.nn as nn
import torch.nn.functional as tf
from torchvision.models import resnet50, vgg16_bn
import networks.blocks as block

class OCR_Segmenter(nn.Module):
    def __init__(self, bottleneck, output_size=1, dropout = 0.5, BN=False):
        super().__init__()
        self.encoder = MNIST_conv(BN)
        self.fc_layer = block.fc_layer(bottleneck, layer_size=[512, 128, output_size],
                                       activation=nn.Softmax, batch_norm=BN)
        #self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, load_size):
        emb= torch.cat([self.encoder(x, _) for _ in load_size], dim=2)
        #x = x.view(x.size(0), -1)
        emb = emb.view(emb.size(0), -1)
        emb = self.dropout(self.fc_layer(emb))
        #return emb
        #y = y / y.norm(2)
        #y = self.localization(emb)
        return self.softmax(emb)

class MNIST_conv(nn.Module):
    def __init__(self, BN=False):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = block.InceptionBlock(1, filters=[[16], [16]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block1", batch_norm=BN)
        self.block1_1 = block.conv_block(32, [32], 1, [1], [1], padding=[0], groups=[1],
                                       name="attention_block_1", batch_norm=BN)
        self.block2 = block.InceptionBlock(32, filters=[[32], [32]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block2", batch_norm=BN)
        self.block2_1 = block.conv_block(64, [64], 1, [1], [1], padding=[0], groups=[1],
                                         name="attention_block_2", batch_norm=BN)
        self.block3 = block.InceptionBlock(64, filters=[[64, 64], [64, 64], [64, 64]],
                                           kernel_sizes=[[(5, 3), 1], [5, 1], [3, 1]],
                                           padding=[[(2, 1), 0], [2, 0], [1, 0]],
                                           stride=[[1, 1], [1, 1], [1, 1]], inner_groups=3, name="incep_block3")
        self.block4 = block.conv_block(192, filters=[192], repeat=1, kernel_sizes=[1],
                                       stride=[1], padding=[0], groups=[1],
                                       name="concat_block", batch_norm=BN)
        """
        self.block1 = block.InceptionBlock(1, filters=[[20], [20]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block1", batch_norm=BN)
        self.block1_1 = block.conv_block(40, [40], 1, [1], [1], padding=[0], groups=[1],
                                       name="attention_block_1", batch_norm=BN)
        self.block2 = block.InceptionBlock(40, filters=[[20], [20]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block2", batch_norm=BN)
        self.block2_1 = block.conv_block(40, [40], 1, [1], [1], padding=[0], groups=[1],
                                         name="attention_block_2", batch_norm=BN)
        self.block3 = block.InceptionBlock(40, filters=[[20], [20]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block3", batch_norm=BN)
        self.block4 = block.conv_block(40, [40], 1, [1], [1], padding=[0], groups=[1],
                                       name="concat_block", batch_norm=BN)
        
        self.block1 = block.conv_block(1, filters=[32, 48], repeat=1, kernel_sizes=[5, 5],
                                       stride=[1, 1], padding=[0, 0], groups=[1, 1], name="block1", batch_norm=BN)
        self.block2 = block.conv_block(48, filters=[48, 48, 64, 64], repeat=1, kernel_sizes=[3, 1, 3, 1],
                                       stride=[1] * 4, padding=[0] * 4, groups=[1] * 4, name="block2", batch_norm=BN)
        self.block3 = block.conv_block(64, filters=[128, 128], repeat=1, kernel_sizes=[3, 1],
                                       stride=[1, 1], padding=[1, 0], groups=[1, 1], name="block3", batch_norm=BN)
        
        self.block1 = block.conv_block(1, filters=[32, 32, 64], repeat=1, kernel_sizes=[3, 3, 1],
                                       stride=[1, 1, 1], padding=[1, 1, 0], groups=[1, 1, 1], name="block1")
        self.block2 = block.conv_block(64, filters=[64, 64, 64], repeat=1, kernel_sizes=[3, 3, 1],
                                       stride=[1, 1, 1], padding=[1, 1, 0], groups=[1, 1, 1], name="block2")
        self.block3 = block.conv_block(64, filters=[128, 128, 64], repeat=1, kernel_sizes=[3, 3, 1],
                                       stride=[1, 1, 1], padding=[1, 1, 0], groups=[1, 1, 1], name="block3")
         """


    def forward(self, input, read_size):
        input = input[:, :, : read_size, :]
        #x = self.maxout(self.block1(input))
        x = self.block1_1(self.maxout(self.block1(input)))
        #attention = x[:, :, int(input.size(2)/4):, int(input.size(3)/4):]
        #x = torch.cat([self.maxout(self.block2(x)), attention], dim=1)
        #x = self.maxout(self.block2(x))
        x = self.block2_1(self.maxout(self.block2(x)))
        x = self.block4(self.maxout(self.block3(x)))
        # x = self.maxout(self.block3(x))
        #x = self.block4(x)
        return x

class Amplified_L1_loss(nn.Module):
    def __init__(self, amplifier=2):
        super().__init__()
        self.amplifier = amplifier

    def forward(self, x, y):
        loss = torch.abs(x - y).pow(1/self.amplifier)
        return loss

if __name__ == "__main__":
    x = torch.rand(32, 1, 32, 40)
    net = MNIST_conv()
    y = net(x, 40)
    print(y.shape)