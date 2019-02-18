import torch
import torch.nn as nn
import omni_torch.networks.blocks as omth_blocks

class CifarNet_Vanilla(nn.Module):
    def __init__(self):
        super(CifarNet_Vanilla, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_025 = nn.Dropout2d(0.25)
        self.dropout_050 = nn.Dropout(0.50)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.fc_layer1 = nn.Linear(2304, 512)
        self.fc_layer2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.pool(self.relu(self.conv1_2(x)))
        x = self.dropout_025(x)
        x = self.relu(self.conv2_1(x))
        x = self.pool(self.relu(self.conv2_2(x)))
        x = self.dropout_025(x)

        x = x.view(x.size(0), -1)
        x = self.dropout_050(self.fc_layer1(x))
        #x = self.softmax(self.fc_layer2(x))
        x = self.fc_layer2(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_050 = nn.Dropout(0.50)
        self.conv_block1 = omth_blocks.Conv_Block(input=3, filters=[32, 32], kernel_sizes=[3, 3], stride=[1, 1],
                                                  padding=[1, 0], batch_norm=None, dropout=[0, 0.25])
        self.conv_block2 = omth_blocks.Conv_Block(input=32, filters=[64, 64], kernel_sizes=[3, 3], stride=[1, 1],
                                                  padding=[1, 0], batch_norm=None, dropout=[0, 0.25])
        self.fc_layer = omth_blocks.fc_layer(2304, [512, 10], activation=[nn.ReLU(), None], batch_norm=False)

    def forward(self, x):
        x = self.pool(self.conv_block1(x))
        x = self.pool(self.conv_block2(x))

        x = x.view(x.size(0), -1)
        x = self.dropout_050(self.fc_layer1(x))
        #x = self.softmax(self.fc_layer2(x))
        x = self.fc_layer2(x)
        return x