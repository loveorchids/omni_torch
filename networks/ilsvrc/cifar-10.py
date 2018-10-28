import torch
import torch.nn as tn
import torch.nn.functional as tf
from networks.blocks import Block
from data.special.cifar10 import Cifar10Data

class CifarNet(Block):
    def __init__(self, args, data):
        super(CifarNet, self).__init__()
        self.args = args
        self.data = data
        #self.criterion = tn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
    
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
    
    def network(self, input):
        net = Block.resnet_block(input, filters=[64, 64], kernel_sizes=[3, 3],
                                 stride=[1, 1], padding=[1, 1], repeat=2,
                                 groups=[1, 1])
        net = Block.pooling(net, tn.MaxPool2d)
        net = Block.resnet_block(net, filters=[128, 128], kernel_sizes=[3, 3],
                                 stride=[1, 1], padding=[1, 1], repeat=2,
                                 groups=[1, 1])
        net = Block.pooling(net, tn.MaxPool2d)
        net = Block.resnet_block(net, filters=[256, 256], kernel_sizes=[3, 3],
                                 stride=[1, 1], padding=[1, 1], repeat=2,
                                 groups=[1, 1])
        net = Block.pooling(net, tn.MaxPool2d)
        net = Block.resnet_block(net, filters=[512, 512], kernel_sizes=[3, 3],
                                 stride=[1, 1], padding=[1, 1], repeat=2,
                                 groups=[1, 1])
        net = Block.pooling(net, tn.MaxPool2d)
        net = Block.fc_layer(net, layer_size=[10], activation=None, batch_norm=False)
        return net
    

def test(args):
    net = CifarNet(args, Cifar10Data)
    y = net(torch.randn(8,3,32,32))
    print(y.size())
    
if __name__ is "__main__":
    
    from options.base_options import BaseOptions
    args = BaseOptions().initialize()
    test(args)
    cifarNet = CifarNet(args, Cifar10Data)
    cifarNet.fit()