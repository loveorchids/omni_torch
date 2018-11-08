import os, time, sys
sys.path.append(os.path.expanduser("~/Documents/omni_torch/"))
import multiprocessing as mpi
import torch, cv2
import torch.nn as nn
import torch.nn.functional as tf
from torchvision.models import resnet18, vgg16_bn
import torchvision.transforms as T
import numpy as np
import networks.blocks as block
import networks.img2img.buddha_net_misc as misc
from data.set_img2img import Img2Img_Dataset
from data.set_arbitrary import  Arbitrary_Dataset
import data.data_loader as loader
import data.path_loader as mode
import data


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
        #self.conv_block4 = nn.Sequential(*net[9:12])
        #self.conv_block4.required_grad = False
        #self.conv_block5 = nn.Sequential(*net[12:15])
        #self.conv_block5.required_grad = False
    
    def forward(self, x):
        #assert len(layers) == len(keys)
        # In this scenario, input x is a grayscale image
        x = x.repeat(1, 3, 1, 1)
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        #out4 = self.conv_block4(out3)
        #out5 = self.conv_block5(out4)
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
    
    def forward(self, input):
        out = self.down_conv1(input)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out


def fit(net, evaluator, args, dataset_1, dataset_2, device_1, device_2, optimizer, criterion, finetune=False,
        do_validation=True, validate_every_n_epoch=1):
    net.train()
    model_path = os.path.expanduser(args.latest_model)
    if finetune and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    P_MSE = []
    S_MSE_1 = []
    S_MSE_2 = []
    S_MSE_3 = []
    loss_weight = dict(zip(["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                              [0.65, 0.20, 0.10, 0.05]))
    loss_weight_range = dict(zip(["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                           [(0.30, 1.00), (0.05, 0.50), (0.05, 0.50), (0.04, 0.40)]))
    for epoch in range(args.epoch_num):
        # The mean of curr_epoch is to increase the diversity during deterministic training
        # see code in set_arbitrary.py  => def __getitem__(self, index)
        args.curr_epoch = epoch
        L = []
        start_time = time.time()
        for batch_idx, (img_batch, label_batch) in enumerate(dataset_1):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            p_mse = criterion(prediction, label_batch) * loss_weight["p_mse"]
            p_mse.backward(retain_graph=True)
            if args.S_MSE:
                ps1, ps2, ps3 = evaluator(prediction)
                ls1, ls2, ls3 = evaluator(label_batch)
                #del prediction, label_batch
                s_mse_1 = criterion(ps1, ls1)/ps1.nelement() * loss_weight["s_mse_1"]
                #del ps1, ls1
                s_mse_2 = criterion(ps2, ls2)/ps2.nelement() * loss_weight["s_mse_2"]
                #del ps2, ls2
                s_mse_3 = criterion(ps3, ls3)/ps3.nelement() * loss_weight["s_mse_3"]
                #del ps3, ls3
                s_mse_1.backward(retain_graph=True)
                s_mse_2.backward(retain_graph=True)
                s_mse_3.backward(retain_graph=True)

                S_MSE_1.append(float((s_mse_1).data) / loss_weight["s_mse_1"])
                S_MSE_2.append(float((s_mse_2).data) / loss_weight["s_mse_2"])
                S_MSE_3.append(float((s_mse_3).data) / loss_weight["s_mse_3"])
                L.append(float((p_mse).data) + float((s_mse_1).data) + float((s_mse_2).data)
                         + float((s_mse_3).data))
            optimizer.step()
            P_MSE.append(float((p_mse).data) / loss_weight["p_mse"])
            L.append(float((p_mse).data))
            if batch_idx % 100 == 0:
                loss_dict = dict(zip(["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                                     [float((p_mse).data), float((s_mse_1).data), float((s_mse_2).data),  float((s_mse_3).data)]))
                misc.vis_image(args, [img_batch, prediction, label_batch],
                             epoch, batch_idx, loss_dict, idx=0)
        print("--- loss: %8f at epoch %04d/%04d, cost %3f seconds ---" %
              (sum(L) / len(L), epoch, args.epoch_num, time.time() - start_time))
        if do_validation and epoch % validate_every_n_epoch == 0:
            pass
            #validation(args, net, val_loader, device, criterion, epoch)
            # test(net, args, data_loader, device)
        if args.S_MSE and epoch is not 0 and (epoch+1) % args.update_n_epoch == 0:
            loss_dis = [np.asarray(P_MSE), np.asarray(S_MSE_1),
                        np.asarray(S_MSE_2), np.asarray(S_MSE_3)]
            loss_weight = misc.update_loss_weight(loss_dis,["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                                             loss_weight, loss_weight_range, args.loss_weight_momentum)
            misc.plot_loss_distribution(loss_dis, ["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                                        args.loss_log, "loss_at_", epoch, loss_weight)
            P_MSE = []
            S_MSE_1 = []
            S_MSE_2 = []
            S_MSE_3 = []
        if epoch is not 0 and (epoch+1) % 100 == 0:
            model_path = os.path.join(args.model_dir, args.code_name + "_epoch_" + str(epoch).zfill(4))
            torch.save(net.state_dict(), model_path)
    return L
            
def validation(net, args, val_dataset, device):
    net.eval()
    model_path = os.path.expanduser(args.latest_model)
    if not os.path.exists(model_path):
        raise FileExistsError("Model does not exist, please check the path of model.")
    net.load_state_dict(torch.load(model_path))
    iter = args.segments[0] * args.segments[0] / args.batch_size
    if iter- int(iter) > 0.001:
        raise FloatingPointError("segmentation is incompatible with batch_size")
    prediction = []
    for batch_idx, (img_batch, label_batch) in enumerate(val_dataset):
        img_batch = img_batch.to(device)
        #label_batch = label_batch.to(device)
        prediction += [_ for _ in net(img_batch).data.to("cpu")]
        if batch_idx is not 0 and (batch_idx+1) % int(iter) is 0:
            misc.save_tensor_to_img(args, prediction, batch_idx+1)

def fetch_data(args, sources):
    data = Img2Img_Dataset(args=args, sources=sources, modes=["path"] * 2,
                           load_funcs=[loader.read_image] * 2, dig_level=[0] * 2)
    data.prepare()
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return data_loader

def milti_layer_input(args, sources):
    def combine_multi_image(args, paths, seed, size, ops):
        imgs = []
        for path in paths:
            imgs.append(loader.read_image(args, path, seed, size, ops))
        return torch.cat(imgs)
    data = Arbitrary_Dataset(args=args, sources=sources, modes=["sep_path", "path"],
                             load_funcs=[combine_multi_image, loader.read_image], dig_level=[0] * 2)
    data.prepare()
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return data_loader

def segmented_input(args, source):
    def to_tensor(args, image, seed, size, ops=None):
        trans = T.Compose([T.ToTensor(), T.Normalize((0,0,0), args.img_std)])
        return trans(image)
    data = Arbitrary_Dataset(args=args, sources=source, modes=[mode.load_img_from_path] * 2,
                             load_funcs=[to_tensor] * 2, dig_level=[0] * 2)
    data.prepare()
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=False, **kwargs)
    return data_loader

if __name__ == "__main__":
    data.ALLOW_WARNING = False
    args = misc.prepare_args()
    if args.deterministic_train:
        torch.manual_seed(args.torch_seed)
        
    device = torch.device("cuda:" + args.gpu_id)
    buddhanet = BuddhaNet()
    buddhanet.to(device)
    
    #evaluator = ResNet18()
    evaluator = Vgg16BN()
    evaluator.to(device)
    
    strong_sup_set = fetch_data(args, ["groupa", "groupb"])
    weak_sup_set = fetch_data(args, ["trainA", "trainB"])
    val_set = segmented_input(args, ["trainA", "trainB"])
    #train_set = milti_layer_input(args, [("groupa", "groupb"), "groupb"])


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(buddhanet.parameters(), lr=args.learning_rate,
                                 weight_decay=1e-5)
    #validation(buddhanet, args, val_set, device)
    fit(buddhanet, evaluator, args, strong_sup_set, weak_sup_set,device, device,
        optimizer, criterion, finetune=args.finetune)
    
