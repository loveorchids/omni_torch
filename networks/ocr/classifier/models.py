import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np
from torchvision.models import resnet50, vgg16_bn
import networks.blocks as block

class Classifier(nn.Module):
    def __init__(self, bottleneck, output_size=16, BN=False, device=None):
        super().__init__()
        self.conv_layer = Li_Hongkai(device)
        self.fc_layer = block.fc_layer(bottleneck, layer_size=[1536, 1024, output_size],
                                       activation=nn.Softmax, batch_norm=BN)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        main = self.conv_layer(input)
        main = main.view(main.size(0), -1)
        main = self.fc_layer(main)
        return main
    
class Multi_Modal(nn.Module):
    def __init__(self, BN=True):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.main = simplified_mnist_type1(BN)
        self.upper_fc = block.fc_layer(4608, layer_size=[2048], name="upper_fc1", activation=nn.Softmax)
        self.aux_block_1 = block.conv_block(1, filters=[16, 16], kernel_sizes=[5, 1], stride=[1, 1], padding=[2, 0],
                                            groups=[1, 1], name="aux_block_1", repeat=1, batch_norm=BN)
        self.aux_block_2 = block.conv_block(16, filters=[32, 32], kernel_sizes=[5, 1], stride=[1, 1], padding=[2, 0],
                                              groups=[1, 1], name="aux_block_2", repeat=1, batch_norm=BN)
        self.aux_block_3 = block.conv_block(32, filters=[64, 64], kernel_sizes=[5, 1], stride=[1, 1], padding=[2, 0],
                                              groups=[1, 1], name="aux_block_3", repeat=1, batch_norm=BN)
        self.aux_1_fc = block.fc_layer(512, layer_size=[1024], name="upper_fc1", activation=nn.Softmax)
        
    def forward(self, x):
        main = self.main(x)
        main = self.upper_fc(main.view(main.size(0), -1))
        aux = self.maxout(self.aux_block_1(x[:, :, :, 16:]))
        aux = self.maxout(self.aux_block_2(aux))
        aux = self.maxout(self.aux_block_3(aux))
        aux = self.aux_1_fc(aux.view(aux.size(0), -1))
        return torch.cat([main, aux], dim=1)
    
class Li_Hongkai(nn.Module):
    def __init__(self, device, BN=True):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.old_block = simplified_mnist_type1(BN)
        self.upper_block_1 = block.conv_block(1, filters=[16], kernel_sizes=[5], stride=[1], padding=[2],
                                              groups=[1], name="upper_block_1", repeat=1, batch_norm=BN)
        self.upper_block_2 = block.conv_block(16, filters=[32], kernel_sizes=[5], stride=[1], padding=[2],
                                              groups=[1], name="upper_block_2", repeat=1, batch_norm=BN)
        self.upper_block_3 = block.conv_block(32, filters=[64], kernel_sizes=[5], stride=[1], padding=[2],
                                              groups=[1], name="upper_block_3", repeat=1, batch_norm=BN)
        self.upper_fc = block.fc_layer(1024, layer_size=[1024], name="upper_fc1", activation=nn.Softmax)
        self.middle_block1_1 = block.conv_block(1, filters=[16], kernel_sizes=[5], stride=[1], padding=[2],
                                              groups=[1], name="middle_block1_1", repeat=1, batch_norm=BN)
        self.middle_block1_2 = block.conv_block(16, filters=[32], kernel_sizes=[5], stride=[1], padding=[2],
                                              groups=[1], name="middle_block1_2", repeat=1, batch_norm=BN)
        self.middle_fc_1 = block.fc_layer(256, layer_size=[256], name="middle_fc_1", activation=nn.Softmax)
        self.middle_block2_1 = block.conv_block(1, filters=[16], kernel_sizes=[5], stride=[1], padding=[2],
                                                groups=[1], name="middle_block2_1", repeat=1, batch_norm=BN)
        self.middle_block2_2 = block.conv_block(16, filters=[32], kernel_sizes=[5], stride=[1], padding=[2],
                                                groups=[1], name="middle_block2_2", repeat=1, batch_norm=BN)
        self.middle_fc_2 = block.fc_layer(256, layer_size=[256], name="middle_fc_2", activation=nn.Softmax)
        self.down_fc = block.fc_layer(256, layer_size=[256], name="middle_fc_2", activation=nn.Softmax)
        self.kirsch = kirsch_weight(device)

    def forward(self, x):
        # Convert input into kirsch feature
        #x = tf.conv2d(x, self.kirsch, padding=1)
        #x = torch.max(x, dim=1)[0].unsqueeze(1)
        # Convert input into kirsch feature

        middle = tf.interpolate(x, size=(32, 8))
        upper = self.maxout(self.upper_block_1(x))
        upper = self.maxout(self.upper_block_2(upper))
        upper = self.maxout(self.upper_block_3(upper))
        #upper = self.old_block(x)
        upper = self.upper_fc(upper.view(upper.size(0), -1))
        middle_1 = self.maxout(self.middle_block1_1(middle[:, :, :16, :]))
        middle_1 = self.maxout(self.middle_block1_2(middle_1))
        middle_1 = self.middle_fc_1(middle_1.view(middle_1.size(0), -1))
        middle_2 = self.maxout(self.middle_block2_1(middle[:, :, 16:, :]))
        middle_2 = self.maxout(self.middle_block2_2(middle_2))
        middle_2 = self.middle_fc_2(middle_2.view(middle_2.size(0), -1))
        #down = tf.conv2d(middle, self.kirsch, padding=1)
        #down = torch.max(down, dim=1)[0].unsqueeze(1)
        #down = self.down_fc(down.view(down.size(0), -1))
        return torch.cat([upper, middle_1, middle_2], dim=1)

class simplified_mnist_type1(nn.Module):
    def __init__(self, BN=True):
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
        self.block3 = block.InceptionBlock(64, filters=[[64, 64], [64, 64]], kernel_sizes=[[(3, 5), 1], [3, 1]],
                                           padding=[[(1, 2), 0], [1, 0]],stride=[[1, 1], [1, 1]], inner_groups=2, name="incep_block3")
        self.block4 = block.conv_block(128, [128, 128], 1, kernel_sizes=[3, 1], stride=[1, 1], padding=[0, 0],
                                         groups=[1, 1], name="block4", batch_norm=BN)

    def forward(self, x):
        x = self.block1_1(self.maxout(self.block1(x)))
        x = self.block2_1(self.maxout(self.block2(x)))
        x = self.block4(self.block3(x))
        return x
    
class simplified_mnist_type2(nn.Module):
    def __init__(self, BN=True):
        super().__init__()
        self.block1 = block.InceptionBlock(1, filters=[[16], [16]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block1", batch_norm=BN)
        self.block1_1 = block.conv_block(32, [32], 1, [1], [1], padding=[0], groups=[1],
                                         name="attention_block_1", batch_norm=BN)
        self.conv_maxout_1= block.conv_block(32, [32], 1, [2], [2], padding=[0], groups=[1],
                                         name="maxout_conv_1", batch_norm=BN)
        self.block2 = block.InceptionBlock(32, filters=[[32], [32]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block2", batch_norm=BN)
        self.block2_1 = block.conv_block(64, [64], 1, [1], [1], padding=[0], groups=[1],
                                         name="attention_block_2", batch_norm=BN)
        self.conv_maxout_2 = block.conv_block(64, [64], 1, [2], [2], padding=[0], groups=[1],
                                              name="maxout_conv_1", batch_norm=BN)
        self.block3 = block.InceptionBlock(64, filters=[[64, 64], [64, 64]], kernel_sizes=[[(3, 5), 1], [3, 1]],
                                           padding=[[(1, 2), 0], [1, 0]],stride=[[1, 1], [1, 1]], inner_groups=2, name="incep_block3")
        self.block4 = block.conv_block(128, [128, 128], 1, kernel_sizes=[3, 1], stride=[1, 1], padding=[0, 0],
                                         groups=[1, 1], name="block4", batch_norm=BN)

    def forward(self, x):
        x = self.block1_1(self.conv_maxout_1(self.block1(x)))
        x = self.block2_1(self.conv_maxout_2(self.block2(x)))
        x = self.block4(self.block3(x))
        return x
    
class simplified_mnist_type3(nn.Module):
    def __init__(self, BN=True):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = block.InceptionBlock(1, filters=[[16], [16]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block1", batch_norm=BN)
        self.block1_1 = block.conv_block(32, [32], 1, [1], [1], padding=[0], groups=[1],
                                         name="attention_block_1", batch_norm=BN)
        self.block2 = block.InceptionBlock(32, filters=[[32], [32], [32]], kernel_sizes=[[5], [3], [1]],
                                           padding=[[2], [1], [0]], stride=[[1], [1], [1]], inner_groups=3,
                                           name="incep_block2", batch_norm=BN)
        self.block2_1 = block.conv_block(96, [96], 1, [1], [1], padding=[0], groups=[1],
                                         name="attention_block_2", batch_norm=BN)
        self.block3 = block.InceptionBlock(96, filters=[[96, 96], [96, 96], [96]], kernel_sizes=[[(3, 5), 1], [3, 1], [1]],
                                           padding=[[(1, 2), 0], [1, 0], [0]],stride=[[1, 1], [1, 1], [1]], inner_groups=3,
                                           name="incep_block3")
        self.block4 = block.conv_block(288, [256, 256], 1, kernel_sizes=[3, 1], stride=[1, 1], padding=[0, 0],
                                         groups=[1, 1], name="block4", batch_norm=BN)

    def forward(self, x):
        x = self.block1_1(self.maxout(self.block1(x)))
        x = self.block2_1(self.maxout(self.block2(x)))
        x = self.block4(self.maxout(self.block3(x)))
        return x

class simplified_mnist_type4(nn.Module):
    def __init__(self, BN=True):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = block.InceptionBlock(1, filters=[[16], [16]], kernel_sizes=[[5], [3]],
                                           padding=[[2], [1]], stride=[[1], [1]], inner_groups=2,
                                           name="incep_block1", batch_norm=BN)
        self.block2_1 = block.InceptionBlock(32, filters=[[32], [32], [32]], kernel_sizes=[[5], [3], [1]],
                                           padding=[[2], [1], [0]], stride=[[1], [1], [1]], inner_groups=3,
                                           name="incep_block2_1", batch_norm=BN)
        self.block2_2 = block.InceptionBlock(32, filters=[[16], [16], [16]], kernel_sizes=[[5], [3], [1]],
                                             padding=[[2], [1], [0]], stride=[[1], [1], [1]], inner_groups=3,
                                             name="incep_block2_2", batch_norm=BN)
        self.block3_1 = block.InceptionBlock(96, filters=[[64, 64], [64, 64]], kernel_sizes=[[(3, 5), 1], [3, 1]],
                                           padding=[[(1, 2), 0], [1, 0]], stride=[[1, 1], [1, 1]], inner_groups=2,
                                           name="incep_block3_1")
        self.block3_2 = block.InceptionBlock(48, filters=[[32, 32], [32, 32]], kernel_sizes=[[(3, 5), 1], [3, 1]],
                                           padding=[[(1, 2), 0], [1, 0]], stride=[[1, 1], [1, 1]], inner_groups=2,
                                           name="incep_block3_2")
        self.block4_1 = block.conv_block(128, [64, 64], 1, kernel_sizes=[3, 1], stride=[1, 1], padding=[1, 0],
                                       groups=[1, 1], name="block4_1", batch_norm=BN)
        self.block4_2 = block.conv_block(64, [64], 1, kernel_sizes=[1], stride=[1], padding=[0],
                                       groups=[1], name="block4_2", batch_norm=BN)

    def forward(self, x1, x2):
        x1 = self.block2_1(self.maxout(self.block1(x1)))
        x2 = self.block2_2(self.maxout(self.block1(x2)))

        x1 = self.block4_1(self.maxout(self.block3_1(x1)))
        x2 = self.block4_2(self.maxout(self.block3_2(x2)))

        x1=self.maxout(x1)
        x2=self.maxout(x2)
        return x1, x2


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
        self.block3 = block.InceptionBlock(64, filters=[[64, 64], [32, 32], [64, 64], [32, 32]],
                                           kernel_sizes=[[(5, 3), 1], [5, 1], [3, 1], [(3, 5), 1]],
                                           padding=[[(2, 1), 0], [2, 0], [1, 0], [(1, 2), 0]],
                                           stride=[[1, 1], [1, 1], [1, 1], [1, 1]], inner_groups=4, name="incep_block3")
        self.block4 = block.conv_block(192, filters=[192], repeat=1, kernel_sizes=[1],
                                       stride=[1], padding=[0], groups=[1],
                                       name="concat_block", batch_norm=BN)

    def forward(self, x):
        x = self.block1_1(self.maxout(self.block1(x)))
        x = self.block2_1(self.maxout(self.block2(x)))
        x = self.block4(self.maxout(self.block3(x)))
        return x
    
def kirsch_weight(device):
    kernelG1 = np.array([[5, 5, 5],
                         [-3, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG2 = np.array([[5, 5, -3],
                         [5, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[5, -3, -3],
                         [5, 0, -3],
                         [5, -3, -3]], dtype=np.float32)
    kernelG4 = np.array([[-3, -3, -3],
                         [5, 0, -3],
                         [5, 5, -3]], dtype=np.float32)
    kernelG5 = np.array([[-3, -3, -3],
                         [-3, 0, -3],
                         [5, 5, 5]], dtype=np.float32)
    kernelG6 = np.array([[-3, -3, -3],
                         [-3, 0, 5],
                         [-3, 5, 5]], dtype=np.float32)
    kernelG7 = np.array([[-3, -3, 5],
                         [-3, 0, 5],
                         [-3, -3, 5]], dtype=np.float32)
    kernelG8 = np.array([[-3, 5, 5],
                         [-3, 0, 5],
                         [-3, -3, -3]], dtype=np.float32)
    weight = np.stack([kernelG1, kernelG2, kernelG3, kernelG4, kernelG5, kernelG6, kernelG7, kernelG8], axis=0) / 15
    weight = np.expand_dims(weight, axis=1)
    weight = torch.tensor(weight)
    return weight.to(device)

if __name__ == "__main__":
    x = torch.rand(32, 1, 32, 40)
    net = Classifier()
    y = net(x, 40)
    print(y.shape)