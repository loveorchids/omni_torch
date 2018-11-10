import os, time, sys
sys.path.append(os.path.expanduser("~/Documents/omni_torch/"))
import multiprocessing as mpi
import torch
import torch.nn as nn
import torch.nn.functional as tf
import torchvision.transforms as T
import numpy as np
import networks.img2img.buddha_net.buddha_net_misc as misc
import networks.img2img.buddha_net.buddha_net_model as model
import networks.img2img.buddha_net.buddha_net_preset as preset
from options.base_options import BaseOptions
from data.set_img2img import Img2Img_Dataset
from data.set_arbitrary import  Arbitrary_Dataset
import data.path_loader as mode
import data.data_loader as loader
import data


#  PYCHARM DEBUG
#  --code_name debug --cover_exist --loading_threads 0

# TERMINAL RUN  (Fill the --code_name)
"""
python3 ~/Documents/omni_torch/networks/img2img/buddha_net  --gpu_id 1 \
--loading_threads 2  --model1 buddanet_recur --model2 vgg16bn \
--general_options 01 --unique_options 01 --code_name DynaLoss_Recurrent
"""

def fit(net, evaluator, args, dataset_1, dataset_2, optimizer, criterion, finetune=False):
    net.train()
    model_path = os.path.expanduser(args.latest_model)
    if finetune and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    L, P_MSE, S_MSE = [], [], [[], [], []]
    loss_name = args.loss_name
    loss_weight = dict(zip(loss_name, args.loss_weight))
    loss_weight_range = dict(zip(loss_name, args.loss_weight_range))
    for epoch in range(args.epoch_num):
        args.curr_epoch = epoch
        start_time = time.time()
        for batch_idx, (img_batch, label_batch) in enumerate(dataset_1):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            if type(prediction) is list:
                p_mse = [criterion(pred, label_batch) * loss_weight["p_mse"]/len(prediction) for pred in prediction]
                for p in p_mse:
                    p.backward(retain_graph=True)
                p_mse = sum(p_mse)
            else:
                p_mse = criterion(prediction, label_batch) * loss_weight["p_mse"]
                p_mse.backward(retain_graph=True)
            P_MSE.append(float(p_mse.data) / loss_weight["p_mse"])
            loss_sum = []
            if args.S_MSE:
                pred_sem_loss = evaluator(prediction)
                gt_sem_loss = evaluator(label_batch)
                assert max(len(pred_sem_loss), len(gt_sem_loss), len(loss_name)-1, len(S_MSE)) is \
                       min(len(pred_sem_loss), len(gt_sem_loss), len(loss_name)-1, len(S_MSE))
                s_mse_losses = [criterion(pred_s_mse, gt_sem_loss[i])/pred_s_mse.nelement() * loss_weight[loss_name[i+1]]
                                            for i, pred_s_mse in enumerate(pred_sem_loss)]
                for i, s_mse in enumerate(s_mse_losses):
                    s_mse.backward(retain_graph=True)
                    S_MSE[i].append(float(s_mse.data) / loss_weight[loss_name[i+1]])
                    loss_sum.append(float(s_mse.data))
                L.append(float(p_mse.data) + sum(loss_sum))
            else:
                L.append(float(p_mse.data))
            optimizer.step()

            if batch_idx % 100 == 0:
                loss_dict = dict(zip(loss_name, [float(p_mse.data)] + loss_sum))
                if type(prediction) is list:
                    prediction = prediction[-1]
                misc.vis_image(args, [img_batch, prediction, label_batch],
                             epoch, batch_idx, loss_dict, idx=0)
        print("--- loss: %8f at epoch %04d/%04d, cost %3f seconds ---" %
              (sum(L) / len(L), epoch, args.epoch_num, time.time() - start_time))
        
        #del s_mse_losses, s_mse, prediction, label_batch, pred_sem_loss, gt_sem_loss
        if args.S_MSE and epoch is not 0 and epoch % args.update_n_epoch == 0:
            # Visualize the gradient
            """
            for name, param in net.named_parameters():
                if len(param.shape) != 4:
                    continue
                print("Visualizing gradient of layer: " + name)
                title = "Loss on: " + name
                sub_title = "Loss norm in L1: %6f L2: %6f" %(float(param.grad.norm(1)), float(param.grad.norm(2)))
                img_path = os.path.join(args.grad_log, name + "_" + str(epoch).zfill(4) + ".jpg")
                misc.plot(param.grad, op=misc.normalize_grad_to_image, title=title, sub_title=sub_title, path=img_path)
            print("Gradient visualization completed")
            """
            loss_dis = [np.asarray(P_MSE)] + [np.asarray(_) for _ in S_MSE]
            loss_weight = misc.update_loss_weight(loss_dis,loss_name, loss_weight, loss_weight_range, args.loss_weight_momentum)
            misc.plot_loss_distribution(loss_dis, loss_name, args.loss_log, "loss_at_", epoch, loss_weight)
            P_MSE, S_MSE = [], [[], [], []]
        if epoch is not 0 and epoch % 50 == 0:
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
# =========LOADING DATASET ==========

def prepare_args():
    def verify_existence(path, raiseError):
        if not os.path.exists(path):
            os.mkdir(path)
        elif not raiseError:
            raise FileExistsError("such code name already exists")
    args = BaseOptions().initialize()
    args.path = os.path.expanduser("~/Pictures/dataset/buddha")
    args.model_dir = os.path.join(args.path, "models")
    args.log_dir = os.path.join(args.path, args.code_name, "log")
    args.loss_log = os.path.join(args.path, args.code_name, "loss")
    args.grad_log = os.path.join(args.path, args.code_name, "grad")
    #  General Options
    try:
        int(args.general_options)
        general_settings = preset.PRESET_Gen[args.general_options]()
        args = general_settings.set(args)
    except ValueError:
        assert os.path.exists(os.path.expanduser(args.general_options))
        args = preset.load_preset(args, args.general_options)
    #  Unique Options
    try:
        int(args.unique_options)
        unique_settings = preset.PRESET_Unq[args.unique_options]()
        args = unique_settings.set(args)
    except ValueError:
        assert os.path.exists(os.path.expanduser(args.unique_options))
        args = preset.load_preset(args, args.general_options)

    verify_existence(os.path.join(args.path, args.code_name), args.cover_exist)
    verify_existence(args.log_dir, args.cover_exist)
    verify_existence(args.loss_log, args.cover_exist)
    verify_existence(args.grad_log, args.cover_exist)
    
    assert args.batch_size == 1, "set batch size to 1 makes the training result better"
    assert args.loss_name[0] == "p_mse"
    assert sum(args.loss_weight) == 1.0
    assert all([True if _[0]>=0 and _[1]<=1 else False for _ in args.loss_weight_range])
    return args

if __name__ == "__main__":
    #net = BuddhaNet_Recur(3).to("cuda:0")
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-5)
    #x = torch.rand(8, 1, 64, 64).to("cuda:0")
    #y = net(x)
    
    data.ALLOW_WARNING = False
    args = prepare_args()
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    device = torch.device("cuda:" + args.gpu_id)
    
    buddhanet = model.MODEL[args.model1.lower()]()
    evaluator = model.MODEL[args.model2.lower()]()
    buddhanet.to(device)
    evaluator.to(device)
    
    strong_sup_set = fetch_data(args, ["groupa", "groupb"])
    #weak_sup_set = fetch_data(args, ["trainA", "trainB"])
    #val_set = segmented_input(args, ["trainA", "trainB"])
    #train_set = milti_layer_input(args, [("groupa", "groupb"), "groupb"])


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(buddhanet.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    #validation(buddhanet, args, val_set, device)
    for epoch in range(args.epoch_num):
        fit(buddhanet, evaluator, args, strong_sup_set, strong_sup_set, optimizer, criterion, finetune=args.finetune)
    
