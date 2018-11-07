import os, time
import torch, cv2
import torch.nn as nn
import torch.nn.functional as tf
from torch.autograd import Variable
from torchvision.models import resnet18, vgg16_bn
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import networks.blocks as block
import networks.img2img.buddha_net_misc as bd_misc

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


def fit(net, evaluator, args, data_loader, val_loader, device, e_device, optimizer, criterion, finetune=False,
        do_validation=True, validate_every_n_epoch=1):
    net.train()
    model_path = os.path.expanduser(args.latest_model)
    if finetune and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    name = ""
    P_MSE = []
    S_MSE_1 = []
    S_MSE_2 = []
    S_MSE_3 = []
    loss_weight = dict(zip(["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                              [0.65, 0.20, 0.1, 0.05]))
    loss_weight_range = dict(zip(["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                           [(0.4, 1), (0.01, 0.5), (0.01, 0.5), (0.01, 0.5)]))
    for epoch in range(args.epoch_num):
        L = []
        start_time = time.time()
        for batch_idx, (img_batch, label_batch) in enumerate(data_loader):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            p_mse = criterion(prediction, label_batch) * loss_weight["p_mse"]
            p_mse.backward(retain_graph=True)
            if args.S_MSE:
                ps1, ps2, ps3 = evaluator(prediction.to(e_device))
                ls1, ls2, ls3 = evaluator(label_batch.to(e_device))
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
                vb.vis_image(args, [img_batch, prediction, label_batch],
                             epoch, batch_idx, L[-1], idx=0)
            #L.append(float((mse_loss).data))
        # os.system("nvidia-smi")
        print("--- loss: %8f at epoch %04d/%04d, cost %3f seconds ---" %
              (sum(L) / len(L), epoch, args.epoch_num, time.time() - start_time))
        if do_validation and epoch % validate_every_n_epoch == 0:
            pass
            #validation(args, net, val_loader, device, criterion, epoch)
            # test(net, args, data_loader, device)
        if args.S_MSE and epoch is not 0 and epoch % 1 == 0:
            loss_dis = [np.asarray(P_MSE), np.asarray(S_MSE_1),
                        np.asarray(S_MSE_2), np.asarray(S_MSE_3)]
            loss_weight = update_loss_weight(loss_dis,["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                                             loss_weight, loss_weight_range, args.loss_weight_momentum)
            plot_loss_distribution(loss_dis, ["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"],
                                   os.path.join(os.path.expanduser(args.path), "loss"),
                                   "loss_at_", epoch, loss_weight)
            
            print(loss_weight)
            P_MSE = []
            S_MSE_1 = []
            S_MSE_2 = []
            S_MSE_3 = []
        if epoch % 100 == 0:
            model_path = os.path.expanduser(args.model)
            torch.save(net.state_dict(), model_path + "_" + str(epoch))
    return

def update_loss_weight(losses, keys, pre, range, momentum=0.8):
    def out_of_bound(num, low_bound, up_bound):
        if num < low_bound:
            # return low bound index
            return 0
        elif num > up_bound:
            # return up bound index
            return 1
        else:
            return -1
    assert len(losses) == len(keys)
    assert momentum >= 0 and momentum < 0.9999
    avg = [float(np.mean(loss)) for loss in losses]
    l = sum(avg)
    weight = [a/l for a in avg]
    avg = [pre[keys[i]] * momentum + weight[i] * (1 - momentum) for i, a in enumerate(avg)]
    print(avg)

    # --------------- support up and low bound of weight -----------------
    mask = [out_of_bound(a, range[keys[i]][0], range[keys[i]][1]) for i, a in enumerate(avg)]
    if mask[0] == -1 and all(mask):
        pass
    else:
        s = sum([avg[i] if n is -1 else 0 for i, n in enumerate(mask)])
        remain = 1 - sum([0 if n is -1 else range[keys[i]][n] for i, n in enumerate(mask)])
        weight = [avg[i] / s * remain if n is -1 else range[keys[i]][n] for i, n in enumerate(mask)]
    # --------------- support up and low bound of weight -----------------
    current = dict(zip(keys, weight))
    return current
    
            
def validation(args, net, val_loader, device, criterion, epoch):
    #model_path = os.path.expanduser(args.latest_model)
    #if os.path.exists(model_path):
        #net.load_state_dict(torch.load(model_path))
    for batch_idx, (img_batch, label_batch) in enumerate(val_loader):
        img_batch = img_batch.to(device).squeeze(0)
        label_batch = label_batch.to(device).squeeze(0)
        prediction = net(img_batch)
        #pred = []
        #for idx in range(img_batch.size(0)):
        #    pred.append(net(img_batch[idx].unsqueeze(0)))
        #prediction = torch.cat(pred, dim=0)
        save_tensor_to_img(args, [1-prediction, label_batch], epoch)

def save_tensor_to_img(args, tensora_list, epoch):
    # -------------------- FOR DEBUG USE ONLY --------------------
    imgs = []
    for tensor in tensora_list:
        vertical_slice = []
        horizontal_slice = []
        for idx in range(tensor.size(0)):
            horizontal_slice.append(tensor[idx].data.to("cpu").numpy().squeeze() * 255)
            if idx is not 0 and (idx+1) % args.segments[1] == 0:
                vertical_slice.append(np.concatenate(horizontal_slice, axis=1))
                horizontal_slice = []
        imgs.append(np.concatenate(vertical_slice, axis=0))
    img = np.concatenate(imgs, axis=1)
    cv2.imwrite(os.path.join(os.path.expanduser(args.log), str(epoch)+".jpg"), img)

def plot_loss_distribution(losses, keyname, save_path, name, epoch, weight):
    names = []
    for key in keyname:
        names.append(key.ljust(8) + ": " + str(weight[key])[:5])
    x_axis = range(len(losses[0]))
    losses.append(np.asarray(list(x_axis)))
    names.append("x")
    plot_data = dict(zip(names, losses))

    # sio.savemat(os.path.expanduser(args.path)+"loss_info.mat", plot_data)
    df = pd.DataFrame(plot_data)
    
    fig, axis = plt.subplots(figsize=(18, 6))
    
    plt.plot("x", names[0], data=df, markersize=1, linewidth=1)
    plt.plot("x", names[1], data=df, markersize=2, linewidth=1)
    plt.plot("x", names[2], data=df, markersize=2, linewidth=1)
    plt.plot("x", names[3], data=df, markersize=3, linewidth=1)
    
    plt.legend(loc='upper right')
    img_name = name + str(epoch).zfill(4) + ".jpg"
    plt.savefig(os.path.join(save_path, img_name))
    #plt.show()


if __name__ == "__main__":
    from options.base_options import BaseOptions
    
    args = BaseOptions().initialize()
    args.path = "~/Pictures/dataset/buddha"
    args.model = "~/Pictures/dataset/buddha/models/train_model"
    args.latest_model = "~/Pictures/dataset/buddha/models/train_model_2100"
    args.log = "~/Pictures/dataset/buddha/logs"
    args.img_channel = 1
    args.batch_size = 1
    args.do_imgaug = False
    args.S_MSE = True
    
    import data
    
    data.ALLOW_WARNING = False
    
    device = torch.device("cuda:1")
    e_device = torch.device("cuda:1")
    buddhanet = BuddhaNet()
    buddhanet.to(device)
    
    #evaluator = ResNet18()
    evaluator = Vgg16BN()
    evaluator.to(e_device)
    
    strong_sup_set = bd_misc.fetch_data(args, ["groupa", "groupb"])
    weak_sup_set = bd_misc.fetch_data(args, ["trainA", "trainB"])
    #train_set = fetch_new_data(args, [("groupa", "groupb"), "groupb"])
    val_set = bd_misc.fetch_val_data(args, ["testA", "testB"])
    for batch_idx, (img_batch, label_batch) in enumerate(val_set):
        img_batch = img_batch.to(device).squeeze(0)
        label_batch = label_batch.to(device).squeeze(0)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(buddhanet.parameters(), lr=args.learning_rate,
                                 weight_decay=1e-5)
    
    fit(buddhanet, evaluator, args, strong_sup_set, val_set,device, e_device,
        optimizer, criterion, finetune=False)
    
    
    
    
    


