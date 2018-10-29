import torch
import torch.nn as nn
import torch.nn.functional as tf
from networks.blocks import ConvBlock
import networks.blocks as block
from data.special.cifar10 import Cifar10Data

class CifarNet(nn.Module):
    def __init__(self, args, data):
        super(CifarNet, self).__init__()
        self.args = args
        self.data = data
        self.create_blocks()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
    
    def fit(self):
        running_loss = 0.0
        CF10 = self.data(self.args, "~/Downloads/cifar-10")
        CF10.prepare()
        for epoch in range(self.args.epoch_num):
            img_batch, label_batch = CF10.get_batch(self.args.batch_size)
            self.optimizer.zero_grad()
            
            prediction = self.resnet_18(img_batch)
            loss = self.criterion(prediction, label_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if epoch % 10 == 0:  # print every 2000 mini-batches
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss / 2000))
                running_loss = 0.0

    def create_blocks(self):
        self.shortcut1 = block.resnet_shortcut(input=3, output=64)
        self.conv_block1 = block.conv_block(input=3, filters=[64, 64], repeat=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.shortcut2 = block.resnet_shortcut(input=64, output=128)
        self.conv_block2 = block.conv_block(input=64, filters=[128, 128],repeat=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.shortcut3 = block.resnet_shortcut(input=128, output=256)
        self.conv_block3 = block.conv_block(input=128, filters=[256, 256], repeat=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.shortcut4 = block.resnet_shortcut(input=256, output=512)
        self.conv_block4 = block.conv_block(input=256, filters=[512, 512], repeat=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.fc_layer = block.fc_layer(input=512, layer_size=[10], activation=None, batch_norm=False)

    
    def forward(self, input):
        res = self.shortcut1(input)
        out = self.conv_block1(input)
        out += res
        out = self.pool1(out)

        res = self.shortcut2(out)
        out = self.conv_block2(out)
        out += res
        out = self.pool2(out)

        res = self.shortcut3(out)
        out = self.conv_block3(out)
        out += res
        out = self.pool3(out)

        res = self.shortcut4(out)
        out = self.conv_block4(out)
        out += res
        out = self.pool4(out)

        out = input.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out
    

def test(args):
    net = CifarNet(args, Cifar10Data)
    y = net(torch.randn(8,3,32,32))
    print(y.size())
    
if __name__ is "__main__":
    from options.base_options import BaseOptions
    args = BaseOptions().initialize()
    #test(args)
    cifarNet = CifarNet(args, Cifar10Data)
    cifarNet.fit()