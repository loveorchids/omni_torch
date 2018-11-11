import torch.nn as nn
import torch.nn.functional as tf
from torchvision.models import resnet18, vgg16_bn
import networks.blocks as block

# ~~~~~~~~~~ NETWORK ~~~~~~~~~~~~
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        # In this scenario, input x is a grayscale image
        x = x.repeat(1, 3, 1, 1)
        for i, layer in enumerate(self.resnet):
            x = layer(x)
        return x


class Vgg16BN(nn.Module):
    def __init__(self):
        super(Vgg16BN, self).__init__()
        vgg16 = vgg16_bn(pretrained=True)
        net = list(vgg16.children())[0]
        self.conv_block1 = nn.Sequential(*net[:3])
        self.conv_block1.required_grad = False
        self.conv_block2 = nn.Sequential(*net[3:6])
        self.conv_block2.required_grad = False
        self.conv_block3 = nn.Sequential(*net[6:9])
        self.conv_block3.required_grad = False
        self.conv_block4 = nn.Sequential(*net[9:12])
        self.conv_block4.required_grad = False
        self.conv_block5 = nn.Sequential(*net[12:15])
        self.conv_block5.required_grad = False
    
    def forward(self, x):
        # assert len(layers) == len(keys)
        # In this scenario, input x is a grayscale image
        x = x.repeat(1, 3, 1, 1)
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        # out4 = self.conv_block4(out3)
        # out5 = self.conv_block5(out4)
        return out1, out2, out3


class BuddhaNet(nn.Module):
    def __init__(self):
        super(BuddhaNet, self).__init__()
        self.down_conv1 = block.conv_block(1, [48, 128, 128], 1, kernel_sizes=[5, 3, 3],
                                           stride=[2, 1, 1], padding=[2, 1, 1], groups=[1] * 3,
                                           activation=nn.SELU, name="block1")
        self.down_conv2 = block.conv_block(128, [256, 256, 256], 1, kernel_sizes=[3, 3, 3],
                                           stride=[2, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block2")
        self.down_conv3 = block.conv_block(256, [256, 512, 1024, 1024, 1024, 1024, 512, 256],
                                           1, kernel_sizes=[3] * 8, stride=[2] + [1] * 7,
                                           padding=[1] * 8, groups=[1] * 8,
                                           activation=nn.SELU, name="block3")
        self.up_conv1 = block.conv_block(256, [256, 256, 128], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block1")
        self.up_conv2 = block.conv_block(128, [128, 128, 48], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block2")
        self.up_conv3 = block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block3")
    
    def forward(self, x):
        out = self.down_conv1(x)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out


class BuddhaNet_MLP(nn.Module):
    def __init__(self):
        super(BuddhaNet_MLP, self).__init__()
        self.down_conv1 = block.conv_block(1, [48, 128, 128], 1, kernel_sizes=[5, 3, 3],
                                           stride=[2, 1, 1], padding=[2, 1, 1], groups=[1] * 3,
                                           activation=nn.SELU, name="block1")
        self.down_conv2 = block.conv_block(128, [256, 256, 256], 1, kernel_sizes=[3, 3, 3],
                                           stride=[2, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block2")
        self.down_conv3 = block.conv_block(256, [256, 512, 1024], 1, kernel_sizes=[3] * 3,
                                           stride=[2] + [1] * 2, padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block3")
        self.inference = block.conv_block(1024, [1024], 1, kernel_sizes=[1],
                                          stride=[1], padding=[0], groups=[1],
                                          activation=nn.SELU, name="inference")
        self.down_conv4 = block.conv_block(1024, [1024, 512, 256], 1, kernel_sizes=[3] * 3,
                                           stride=[1] * 3, padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block4")
        self.up_conv1 = block.conv_block(256, [256, 256, 128], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block1")
        self.up_conv2 = block.conv_block(128, [128, 128, 48], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block2")
        self.up_conv3 = block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block3")
    
    def forward(self, x):
        out = self.down_conv1(x)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.inference(out)
        out = self.down_conv4(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out


class BuddhaNet_NIN(nn.Module):
    def __init__(self):
        super(BuddhaNet_NIN, self).__init__()
        self.down_conv1 = block.conv_block(1, [48, 128, 128, 128], 1, kernel_sizes=[3, 3, 3, 1],
                                           stride=[2, 1, 1, 1], padding=[1, 1, 1, 0], groups=[1] * 4,
                                           activation=nn.SELU, name="block1")
        self.down_conv2 = block.conv_block(128, [256, 256, 256, 256], 1, kernel_sizes=[3, 3, 3, 1],
                                           stride=[2, 1, 1, 1], padding=[1] * 3 + [0], groups=[1] * 4,
                                           activation=nn.SELU, name="block2")
        self.down_conv3 = block.conv_block(256, [256, 512, 1024, 1024], 1, kernel_sizes=[3] * 3 + [1],
                                           stride=[2] + [1] * 3, padding=[1] * 3 + [0], groups=[1] * 4,
                                           activation=nn.SELU, name="block3")
        self.down_conv4 = block.conv_block(1024, [1024, 512, 256, 256], 1, kernel_sizes=[3] * 3 + [1],
                                           stride=[1] * 4, padding=[1] * 3 + [0], groups=[1] * 4,
                                           activation=nn.SELU, name="block4")
        self.up_conv1 = block.conv_block(256, [256, 256, 128, 128], 1, kernel_sizes=[4, 3, 3, 1],
                                         stride=[0.5, 1, 1, 1], padding=[1] * 3 + [0], groups=[1] * 4,
                                         activation=nn.SELU, name="up_block1")
        self.up_conv2 = block.conv_block(128, [128, 128, 48, 48], 1, kernel_sizes=[4, 3, 3, 1],
                                         stride=[0.5, 1, 1, 1], padding=[1] * 3 + [0], groups=[1] * 4,
                                         activation=nn.SELU, name="up_block2")
        self.up_conv3 = block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block3")
    
    def forward(self, x):
        out = self.down_conv1(x)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.down_conv4(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out


class BuddhaNet_NICE(nn.Module):
    def __init__(self, nice_block=5):
        super(BuddhaNet_NICE, self).__init__()
        self.down_conv1 = block.conv_block(1, [48, 128, 128], 1, kernel_sizes=[5, 3, 3],
                                           stride=[2, 1, 1], padding=[2, 1, 1], groups=[1] * 3,
                                           activation=nn.SELU, name="block1")
        self.down_conv2 = block.conv_block(128, [256, 256, 256], 1, kernel_sizes=[3, 3, 3],
                                           stride=[2, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block2")
        self.down_conv3 = block.conv_block(256, [256, 512, 1024], 1, kernel_sizes=[3] * 3,
                                           stride=[2] + [1] * 2, padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block3")
        
        self.up_conv1 = block.conv_block(256, [256, 256, 128], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block1")
        self.up_conv2 = block.conv_block(128, [128, 128, 48], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block2")
        self.up_conv3 = block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block3")
    
    def forward(self, x):
        out = self.down_conv1(x)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.inference(out)
        out = self.down_conv4(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out


class BuddhaNet_Res(nn.Module):
    def __init__(self):
        super(BuddhaNet_Res, self).__init__()
        BATCH_NORM = False
        self.down_conv1 = block.conv_block(1, [48, 128, 128], 1, kernel_sizes=[5, 3, 3],
                                           stride=[2, 1, 1], padding=[2, 1, 1], groups=[1] * 3,
                                           activation=nn.SELU, name="block1", batch_norm=BATCH_NORM)
        self.shortcur1 = block.resnet_shortcut(128, 128, 1, 1, 0, batch_norm=BATCH_NORM)
        self.down_conv2 = block.conv_block(128, [128, 256, 256], 1, kernel_sizes=[3, 3, 3],
                                           stride=[2, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU, name="block2", batch_norm=BATCH_NORM)
        self.shortcur2 = block.resnet_shortcut(256, 256, 1, 1, 0, batch_norm=BATCH_NORM)
        self.down_conv3 = block.conv_block(256, [256, 512, 1024, 1024, 1024, 1024, 512, 256],
                                           1, kernel_sizes=[3] * 8, stride=[2] + [1] * 7,
                                           padding=[1] * 8, groups=[1] * 8,
                                           activation=nn.SELU, name="block3", batch_norm=BATCH_NORM)
        self.shortcur3 = block.resnet_shortcut(256, 256, 1, 1, 0, batch_norm=BATCH_NORM)
        self.up_conv1 = block.conv_block(256, [256, 256, 128], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block1", batch_norm=BATCH_NORM)
        self.shortcur4 = block.resnet_shortcut(128, 128, 1, 1, 0, batch_norm=BATCH_NORM)
        self.up_conv2 = block.conv_block(128, [128, 128, 48], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block2", batch_norm=BATCH_NORM)
        self.shortcur5 = block.resnet_shortcut(48, 48, 1, 1, 0, batch_norm=BATCH_NORM)
        self.up_conv3 = block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                         activation=nn.SELU, name="up_block3", batch_norm=BATCH_NORM)
    
    def forward(self, x):
        res = self.shortcur1(x)
        out = self.down_conv1(x)
        out = tf.relu(out + res)
        
        res = self.shortcur2(out)
        out = self.down_conv2(out)
        out = tf.relu(out + res)
        
        res = self.shortcur3(out)
        out = self.down_conv3(out)
        out = tf.relu(out + res)
        
        res = self.shortcur4(out)
        out = self.up_conv1(out)
        out = tf.relu(out + res)
        
        res = self.shortcur5(out)
        out = self.up_conv2(out)
        out = tf.relu(out + res)
        
        out = self.up_conv3(out)
        return out


class BuddhaNet_Recur(nn.Module):
    def __init__(self):
        super(BuddhaNet_Recur, self).__init__()
        self.batch_norm = False
        self.net1 = nn.Sequential(*self.recurrent_block_big())
        self.net2 = nn.Sequential(*self.recurrent_block_small())
        self.net3 = nn.Sequential(*self.recurrent_block_small())
        
    def recurrent_block_big(self):
        return [
            block.conv_block(1, [48, 128, 128], 1, kernel_sizes=[3, 3, 1], stride=[2, 1, 1], padding=[1, 1, 0],
                             groups=[1] * 3, activation=nn.SELU, name="block1"),
            block.conv_block(128, [256, 256, 256], 1, kernel_sizes=[3, 3, 1], stride=[2, 1, 1],
                             padding=[1] * 2 + [0], groups=[1] * 3, activation=nn.SELU, name="block2"),
            block.conv_block(256, [256, 512, 1024, 1024], 1, kernel_sizes=[3] * 3 + [1], stride=[2] + [1] * 3,
                             padding=[1] * 3 + [0], groups=[1] * 4, activation=nn.SELU, name="block3"),
            block.conv_block(1024, [1024, 512, 256], 1, kernel_sizes=[3] * 2 + [1], stride=[1] * 3,
                             padding=[1] * 2 + [0], groups=[1] * 3, activation=nn.SELU, name="block4"),
            block.conv_block(256, [256, 256, 128], 1, kernel_sizes=[4, 3, 1], stride=[0.5, 1, 1],
                             padding=[1] * 2 + [0], groups=[1] * 3, activation=nn.SELU, name="up_block1"),
            block.conv_block(128, [128, 48, 48], 1, kernel_sizes=[4, 3, 1], stride=[0.5, 1, 1],
                             padding=[1] * 2 + [0], groups=[1] * 3, activation=nn.SELU, name="up_block2"),
            block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1],
                             padding=[1] * 3, groups=[1] * 3, activation=nn.SELU, name="up_block3")]
    
    def recurrent_block_small(self):
        return [
            # Down Conv Layer
            block.conv_block(1, [64, 128, 256], 1, kernel_sizes=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1],
                             groups=[1] * 3, activation=nn.SELU, name="block1", batch_norm=self.batch_norm),
            # Inference Layer
            block.conv_block(256, [256], 2, kernel_sizes=[1], stride=[1], padding=[0], groups=[1],
                             activation=nn.SELU, batch_norm=self.batch_norm),
            # Up Conv Layer
            block.conv_block(256, [256, 128, 64, 1], 1, kernel_sizes=[4, 4, 4, 1], stride=[0.5, 0.5, 0.5, 1],
                             padding=[1, 1, 1, 0], groups=[1] * 4, activation=nn.SELU, name="up_block1",
                             batch_norm=self.batch_norm)
        ]
    
    def forward(self, x):
        result = []
        x = self.net1(x)
        result.append(x)
        x = self.net2(x)
        result.append(x)
        x = self.net3(x)
        result.append(x)
        return result
# ~~~~~~~~~~ NETWORK ~~~~~~~~~~~~

MODEL = {
    "resnet18": ResNet18,
    "vgg16bn": Vgg16BN,
    "buddhanet_mlp": BuddhaNet_MLP,
    "buddhanet_res": BuddhaNet_Res,
    "buddhanet_nice": BuddhaNet_NICE,
    "buddhanet_nin": BuddhaNet_NIN,
    "buddhanet_recur": BuddhaNet_Recur,
    "buddhanet": BuddhaNet
}