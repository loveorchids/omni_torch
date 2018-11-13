import os, time, sys, math
sys.path.append(os.path.expanduser("~/Documents/omni_torch/"))
import multiprocessing as mpi
import torch
import torch.nn as nn
import torch.nn.functional as tf
import torchvision.transforms as T
import numpy as np
import visualize.basic as vb
import networks.util as util
import networks.img2img.buddha_net as basic
import networks.img2img.buddha_net.models as model
import networks.img2img.buddha_net.presets as preset
from options.base_options import BaseOptions
from data.set_img2img import Img2Img_Dataset
from data.set_arbitrary import Arbitrary_Dataset
import data.path_loader as mode
import data.data_loader as loader
import data

#  PYCHARM DEBUG
#  --code_name debug --cover_exist --loading_threads 0

# TERMINAL RUN  (Fill the --code_name)
"""
python3 ~/Documents/omni_torch/networks/img2img/buddha_net  --gpu_id 1 \
--loading_threads 2  --model1 buddhanet_nin --model2 vgg16bn \
--general_options 01 --unique_options 01 --code_name DynaLoss_NIN
"""


def fit(net, evaluator, args, dataset_1, dataset_2, device, optimizer, criterion, finetune=False):
    net.train()
    if finetune:
        net = util.load_latest_model(args, net)
    # =============== Special Part ================
    L, P_MSE, S_MSE = [], [], [[], [], []]
    loss_name = args.loss_name
    # =============== Special Part ================
    for epoch in range(args.update_n_epoch):
        args.curr_epoch += 1
        start_time = time.time()
        for batch_idx, (img_batch, label_batch) in enumerate(dataset_1):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            if type(prediction) is list:
                p_mse = [criterion(pred, label_batch) * args.loss_weight["p_mse"] / len(prediction) for pred in prediction]
                for p in p_mse:
                    p.backward(retain_graph=True)
                p_mse = sum(p_mse)
            else:
                p_mse = criterion(prediction, label_batch) * args.loss_weight["p_mse"]
                p_mse.backward(retain_graph=True)
            P_MSE.append(float(p_mse.data) / args.loss_weight["p_mse"])
            loss_sum = []
            # =============== Special Part ================
            if args.S_MSE:
                if type(prediction) is list:
                    pred_sem_loss = evaluator(prediction[-1])
                else:
                    pred_sem_loss = evaluator(prediction)
                gt_sem_loss = evaluator(label_batch)
                assert max(len(pred_sem_loss), len(gt_sem_loss), len(loss_name) - 1, len(S_MSE)) is \
                       min(len(pred_sem_loss), len(gt_sem_loss), len(loss_name) - 1, len(S_MSE))
                s_mse_losses = [
                    criterion(pred_s_mse, gt_sem_loss[i]) / pred_s_mse.nelement() * args.loss_weight[loss_name[i + 1]]
                    for i, pred_s_mse in enumerate(pred_sem_loss)]
                for i, s_mse in enumerate(s_mse_losses):
                    s_mse.backward(retain_graph=True)
                    S_MSE[i].append(float(s_mse.data) / args.loss_weight[loss_name[i + 1]])
                    loss_sum.append(float(s_mse.data))
                L.append(float(p_mse.data) + sum(loss_sum))
            # =============== Special Part ================
            else:
                L.append(float(p_mse.data))
            optimizer.step()
            
            # Visualize Output and Loss
            if batch_idx % 100 == 0:
                loss_dict = dict(zip(loss_name, [float(p_mse.data)] + loss_sum))
                if type(prediction) is list:
                    prediction = prediction[-1]
                img_name = str(args.curr_epoch) + "_" + str(batch_idx) + ".jpg"
                path = os.path.join(os.path.expanduser(args.log_dir), img_name)
                description = ""
                keys = sorted(loss_dict.keys())
                for _ in keys:
                    description += _ + " loss: %6f " %(loss_dict[_])
                vb.plot_tensor(torch.cat([img_batch, prediction, label_batch]), path, title=description,
                          deNormalize=True, font_size = 0.7)
        print("--- loss: %8f at epoch %04d/%04d, cost %3f seconds ---" %
              (sum(L) / len(L), epoch, args.update_n_epoch, time.time() - start_time))
        # del s_mse_losses, s_mse, prediction, label_batch, pred_sem_loss, gt_sem_loss
        
    start = time.time()
    print("Starting to visualize gradient")
    vb.visualize_gradient(args, net)
    print("Gradient visualization completed, took %3f second" % (time.time() - start))
    # =============== Special Part ================
    if args.S_MSE:
        loss_dis = [np.asarray(P_MSE)] + [np.asarray(_) for _ in S_MSE]
        args.loss_weight = basic.update_loss_weight(loss_dis, loss_name, args.loss_weight, args.loss_weight_range,
                                              args.loss_weight_momentum)
        basic.plot_loss_distribution(loss_dis, loss_name, args.loss_log, "loss_at_", args.curr_epoch, args.loss_weight)
    # =============== Special Part ================
    return L


def validation(net, args, val_dataset, device):
    net.eval()
    net = util.load_latest_model(args, net)
    iter = args.segments[0] * args.segments[0] / args.batch_size
    if iter - int(iter) > 0.001:
        raise FloatingPointError("segmentation is incompatible with batch_size")
    prediction = []
    for batch_idx, (img_batch, label_batch) in enumerate(val_dataset):
        img_batch = img_batch.to(device)
        pred = net(img_batch)
        if type(pred) is list:
            prediction += [_ for _ in pred[-1].data]
        # label_batch = label_batch.to(device)
        else:
            prediction += [_ for _ in pred.data]
        if batch_idx is not 0 and (batch_idx + 1) % int(iter) is 0:
            prediction = torch.cat(prediction, dim=0).unsqueeze(1)
            path = os.path.join(os.path.expanduser(args.log_dir), "val_" + str(args.curr_epoch)+".jpg")
            vb.plot_tensor(prediction, path, margin=0, deNormalize=True)
            prediction = []


# =========LOADING DATASET ==========
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
        trans = T.Compose([T.ToTensor(), T.Normalize((0, 0, 0), args.img_std)])
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

# =========LOADING DATASET ==========

def verify_args(args, presets):
    args = util.prepare_args(args, presets)

    args.loss_weight = dict(zip(args.loss_name, args.loss_weight))
    args.loss_weight_range = dict(zip(args.loss_name, args.loss_weight_range))

    assert args.batch_size == 1, "set batch size to 1 makes the training result better"
    assert args.loss_name[0] == "p_mse"
    assert 0.9999 <= sum([_ for _ in args.loss_weight.values()]) <= 1.0001
    assert all([True if _[0] >= 0 and _[1] <= 1 else False for _ in args.loss_weight_range.values()])
    return args


if __name__ == "__main__":
    data.ALLOW_WARNING = False
    args = BaseOptions().initialize()
    
    # Running Info
    args.loading_threads = 2
    args.model1 = "buddhanet_unet"
    args.model2 = "vgg16bn"
    args.general_options = "G_01"
    args.unique_options = "U_01"
    args.cover_exist = True
    args.code_name = "Dynaloss_Unet_var1"
    args.gpu_id = "0"
    args = verify_args(args, preset.PRESET)
    
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    device = torch.device("cuda:" + args.gpu_id)
    
    #x = torch.rand((1, 1, 352, 352))
    buddhanet = model.MODEL[args.model1.lower()]()
    #y = buddhanet(x)
    evaluator = model.MODEL[args.model2.lower()]()
    buddhanet.to(device)
    evaluator.to(device)
    
    strong_sup_set = fetch_data(args, ["groupa", "groupb"])
    # weak_sup_set = fetch_data(args, ["trainA", "trainB"])
    val_set = segmented_input(args, ["valA", "valB"])
    # train_set = milti_layer_input(args, [("groupa", "groupb"), "groupb"])
    
    criterion = nn.MSELoss()
    

    for epoch in range(args.epoch_num):
        print("=====================PHASE: %s=====================" %(str(epoch).zfill(3)))
        optimizer = torch.optim.Adam(buddhanet.parameters(), lr=args.learning_rate * math.pow(0.9, epoch),
                                     weight_decay=args.weight_decay)
        fit(buddhanet, evaluator, args, strong_sup_set, strong_sup_set, device, optimizer, criterion, finetune=args.finetune)
        util.save_model(args, epoch, buddhanet.state_dict())
        validation(buddhanet, args, val_set, device)