import os, time
import torch, cv2
import torch.nn as nn
import torch.nn.functional as tf
from torch.autograd import Variable
from torchvision.models import resnet18
import networks.blocks as block
from data.img2img import Img2Img
import data.data_loader as loader
import visualize.basic as vb


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        for i, layer in enumerate(self.resnet):
            x = layer(x)
        return x


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
    
    def forward(self, input):
        out = self.down_conv1(input)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out
    
    def semantic(self, input):
        out = self.down_conv1(input)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = tf.max_pool2d(out, 2, 2)
        return out


def fit(net, evaluator, args, data_loader, device, optimizer, criterion, finetune=False,
        do_validation=True, validate_every_n_epoch=5):
    net.train()
    model_path = os.path.expanduser(args.model)
    if finetune and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    for epoch in range(args.epoch_num):
        L = []
        start_time = time.time()
        for batch_idx, (img_batch, label_batch) in enumerate(data_loader):
            
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            pred_sementic = evaluator(prediction.to("cuda:0"))
            label_sementic = evaluator(label_batch.to("cuda:0"))
            if batch_idx % 100 == 0:
                vb.vis_image(args, [img_batch, prediction, label_batch],
                             epoch, batch_idx, idx=0)
                # pass
            mse_loss = criterion(prediction, label_batch)
            loss = criterion(pred_sementic, label_sementic)*0.1
            
            mse_loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            
            optimizer.step()
            
            L.append(float((mse_loss).data) + float((loss).data))
            #L.append(float((mse_loss).data))
        # os.system("nvidia-smi")
        print("--- loss: %8f at epoch %04d, cost %3f seconds ---" %
              (sum(L) / len(L), epoch, time.time() - start_time))
        if do_validation and epoch % validate_every_n_epoch == 0:
            pass
            # test(net, args, data_loader, device)
        if epoch % 100 == 0:
            model_path = os.path.expanduser(args.model)
            torch.save(net.state_dict(), model_path + "_" + str(epoch))


def save_tensor_to_img(tensor, idx):
    img = tensor[idx].data.to("cpu").numpy().squeeze() * 255
    cv2.imwrite(os.path.expanduser("~/Pictures/tmp.jpg"), img)


def fetch_data(args, sources):
    import multiprocessing as mpi
    data = Img2Img(args=args, sources=sources, modes=["path"] * 2,
                   load_funcs=[loader.read_image] * 2, dig_level=[0] * 2)
    data.prepare()
    works = mpi.cpu_count() - 2
    kwargs = {'num_workers': works, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return data_loader


if __name__ == "__main__":
    from options.base_options import BaseOptions
    
    args = BaseOptions().initialize()
    args.path = "~/Pictures/dataset/buddha"
    args.model = "~/Pictures/dataset/buddha/models/train_model"
    args.log = "~/Pictures/dataset/buddha/logs"
    args.img_channel = 1
    args.batch_size = 1
    
    import data
    
    data.ALLOW_WARNING = False
    
    device = torch.device("cuda:1")
    buddhanet = BuddhaNet()
    buddhanet.to(device)
    
    resnet18 = ResNet18()
    resnet18.to("cuda:0")
    
    train_set = fetch_data(args, ["groupa", "groupb"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(buddhanet.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    fit(buddhanet, resnet18, args, train_set, device, optimizer, criterion, finetune=False)


