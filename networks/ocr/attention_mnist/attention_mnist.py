import os, time, sys, math, random

sys.path.append(os.path.expanduser("~/Documents/omni_research/"))

import cv2, torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np
import visualize.basic as vb
import networks.util as util
import networks.loss as loss
import networks.ocr.attention_mnist.misc as misc
import networks.ocr.attention_mnist.models as model
import networks.ocr.attention_mnist.presets as preset
from options.base_options import BaseOptions

TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
invert_label_ori = {12: '-', 3: '¥', 11: '0', 2: ',', 7: '〒', 6: '3', 5: '4', 8: '7', 14: '5', 0: '.', 9: '1',
                    13: '8', 4: '6', 1: '9', 10: '2', 15: '', 16: "/", 17: "(", 18: ")"}


# Treat dot as comma, see key 0 and 2


def plot_seperator(img_batch, sep_batch):
    pred_imgs = []
    for i in range(sep_batch.size(0)):
        pred_img = misc.draw_line(img_batch[i].unsqueeze(0), gt=int(sep_batch[i]))
        pred_imgs.append(pred_img)
    vb.plot_tensor(torch.cat(pred_imgs, dim=0), TMPJPG, deNormalize=True)


def fit(args, net, train_set, val_set, device, optimizer, criterions, is_train=True):
    if is_train:
        net.train()
        dataset = train_set
        prefix = ""
        iter = args.epoches_per_phase
    else:
        net.eval()
        dataset = val_set
        prefix = "VAL"
        iter = 1
    epoch_Loss = []
    for epoch in range(iter):
        Loss, predictions, pred_imgs, distance = [], [], [], []
        start_time = time.time()
        # for batch_idx, (img_batch, label, num) in enumerate(dataset):
        for batch_idx, data in enumerate(dataset):
            # Get data ops
            img_batch, sep_batch = data[0][0].to(device), data[0][1]
            sep_gaussian = misc.to_gaussian_prob(args.model_load_size[-1], sep_batch, var=1).to(device)
            # one_hot = loader.one_hot(args.model_load_size[-1], int(img[1])-1).unsqueeze(0)
            
            # Prediction and optimization
            pred = net(img_batch, args.model_load_size)
            loss = sum([criterion(pred, sep_gaussian) for criterion in criterions])
            # loss = criterion(pred, sep_gaussian) + criterion(sep_gaussian, pred) + criterion2(pred, sep_gaussian)
            Loss.append(float(loss.data))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Special Visualization Ops
            _, pred_idx = torch.max(pred, dim=1)
            _, sep_idx = torch.max(sep_gaussian, dim=1)
            for i in range(pred_idx.size(0)):
                distance.append(abs(int(pred_idx[i]) - int(sep_idx[i])))
                if abs(int(pred_idx[i]) - int(sep_idx[i])) > args.bad_distance and not is_train:
                    # Very Bad Prediction
                    # print("pred: ", int(pred_idx[i]), " label: ", int(sep_batch[i]))
                    pred_img = misc.draw_line(img_batch[i].unsqueeze(0), int(pred_idx[i]), int(sep_idx[i]))
                    pred_imgs.append(pred_img)
        #
        if pred_imgs and args.curr_epoch != 0:
            path = os.path.expanduser("~/Pictures/%s_tmp_%s.jpg" %
                                      ("Train", str(args.curr_epoch).zfill(4)))
            vb.plot_tensor(torch.cat(pred_imgs, dim=0), path,
                           title="Avg Prediction Error: %s px" % (sum(distance) / len(distance)),
                           ratio=1.6, deNormalize=True, bg_color=224, margin=10)
            pred_imgs.clear()
        print(prefix + " --- loss: %8f, accuracy: %4s, at epoch %04d/%04d, cost %3f seconds ---" %
              (sum(Loss) / len(Loss), str(1 - sum(distance) / len(distance) / args.model_load_size[-1])[:6],
               epoch + 1, args.epoches_per_phase, time.time() - start_time))
        epoch_Loss += Loss
        if is_train:
            args.curr_epoch += 1
    if is_train:
        # start = time.time()
        # vb.visualize_gradient(args, net, minimun_rank=4)
        # print("Gradient visualization completed, took %3f second" % (time.time() - start))
        vb.plot_loss_distribution([epoch_Loss], keyname=None, save_path=os.path.expanduser(args.loss_log),
                                  name="Loss_trend", epoch=args.curr_epoch, weight=None)
        return net


def get_image_slice(img_batch, start, width, device):
    width_end = min(start + width, img_batch.size(3))
    img = img_batch[:, :, :, start: width_end]
    if img.size(3) < width:
        # Make the background as white
        bg = (torch.zeros((1, 1, img.size(2), width)) + 0.49).to(device)
        bg[:, :, :, :img.size(3)] = img
        return bg.to(device)
    else:
        return img.to(device)

def validation(args, net, dataset, device):
    print("Start Validation")
    net.eval()
    with torch.no_grad():
        # for idx, (img, label, coord, num) in enumerate(dataset):
        for idx, img in enumerate(dataset):
            sep = []
            start = 0
            if idx % 100 == 0 and idx != 0:
                print("100 samples completed")
            # coord = [int(_) for _ in coord[0]]
            img = img[0][0]
            while start < img.size(3):
                # while start + args.model_load_size[-1] < img.size(3):
                slice = get_image_slice(img, start, args.model_load_size[-1], device)
                pred = net(slice, args.model_load_size)
                _, pred_idx = torch.max(pred, dim=1)
                start += (int(pred_idx) + 1)
                sep.append(start)
            # pred_img = misc.draw_multi_line(img, coord =sep, gt=coord)
            pred_img = misc.draw_multi_line(img, coord=sep)
            path = os.path.expanduser("~/Pictures/whole_%s_tmp_%s.jpg" % ("VAL", idx))
            vb.plot_tensor(pred_img, path, deNormalize=True, bg_color=224, margin=10)

def verify_args(args, presets, options=None):
    args = util.prepare_args(args, presets, options=options)
    # Verification Code Here
    if args.batch_size == 1:
        args.batch_norm = False
    return args


def main():
    args = BaseOptions().initialize()
    args = verify_args(args, preset.PRESET, options=["ocr1", "U_01", "runtime"])
    
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    device = torch.device("cuda:" + args.gpu_id)
    
    train_set, val_set = misc.fetch_hybrid_data(args, sources=args.img_folder_name,
                                                foldername=args.img_folder_name, shuffle=False,
                                                batch_size=args.batch_size)
    tmp_holding = misc.fetch_data(args, sources=["test_set"], batch_size=1, txt_file=["numbers.txt"])
    
    net = model.OCR_Segmenter(bottleneck=args.bottleneck_size, output_size=args.model_load_size[-1],
                              dropout=0, BN=args.batch_norm).to(device)
    criterion = [loss.JS_Divergence(), nn.MSELoss()]
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.finetune:
        net = util.load_latest_model(args, net)
    for epoch in range(args.epoch_num):
        print("\n ======================== PHASE: %s/%s ========================= " %
              (str(int(args.curr_epoch / args.epoches_per_phase)).zfill(4), str(args.epoch_num).zfill(4)))
        fit(args, net, train_set, val_set, device, optimizer, criterion, is_train=False)
        net = fit(args, net, train_set, val_set, device, optimizer, criterion, is_train=True)
        util.save_model(args, args.curr_epoch, net.state_dict())
        if epoch != 0 and epoch % 5 == 0:
            start = time.time()
            validation(args, net, tmp_holding, device)
            print("Validation took %s seconds" % (time.time() - start))

if __name__ == "__main__":
    main()