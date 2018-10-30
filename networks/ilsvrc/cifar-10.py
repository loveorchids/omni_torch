import torch
import torch.nn as nn
import torch.nn.functional as tf
import networks.blocks as block
from data.arbitrary import Arbitrary
import data.data_loader as loader
import data.mode as mode


class CifarNet(nn.Module):
    def __init__(self, args):
        super(CifarNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.create_blocks()
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
    
    def fit(self):
        running_loss = 0.0
        print("loading Dataset...")
        data = Arbitrary(args=args, load_funcs=[loader.to_tensor, self.just_return_it],
                         sources=[("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4")],
                         modes=[mode.load_pickle_from_cifar], dig_level=[0])
        data.prepare()
        print("loading Completed!")
        
        kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)
        
        for epoch in range(self.args.epoch_num):
            for batch_idx, (img_batch, label_batch) in enumerate(train_loader):
                img_batch, label_batch = img_batch.to(self.device), label_batch.to(self.device)
            
                prediction = self.forward(img_batch)
                loss = self.criterion(prediction, label_batch)
                print("--- loss: %s at batch %d---" % (loss, batch_idx))
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

        self.fc_layer = block.fc_layer(input=3072, layer_size=[10], activation=None, batch_norm=False)

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
    
    @staticmethod
    def just_return_it(args, data, seed=None, size=None):
        """
        Because the label in cifar dataset is int
        So here it will be transfered to a torch tensor
        """
        return torch.tensor(data)

def test(args):
    net = CifarNet(args)
    y = net(torch.randn(8,3,32,32))
    print(y.size())
    
if __name__ is "__main__":
    from options.base_options import BaseOptions
    args = BaseOptions().initialize()
    args.path = "~/Downloads/cifar-10"
    #test(args)
    cifarNet = CifarNet(args)
    cifarNet.fit()