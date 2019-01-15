import os, time, sys, math, random

sys.path.append(os.path.expanduser("~/Documents/omni_research/"))

import cv2, torch
import torch.nn as nn
import numpy as np
import visualize.basic as vb
import networks.util as util
import networks.ocr.classifier.misc as misc
import networks.ocr.attention_mnist.misc as seg_misc
import networks.ocr.classifier.models as model
import networks.ocr.classifier.presets as preset
import networks.ocr.attention_mnist.models as seg_model
import networks.ocr.attention_mnist.presets as seg_preset
import networks.ocr.test as test
from options.base_options import BaseOptions
import data.data_loader as loader

import data

TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
              '-': 12, '8': 13, '5': 14, '': 15, "/": 16, "(": 17, ")": 18}
invert_label = {12: '-', 3: '¥', 11: '0', 2: ',', 7: '〒', 6: '3', 5: '4', 8: '7', 14: '5', 0: '.', 9: '1',
                13: '8', 4: '6', 1: '9', 10: '2', 15: ' ', 16: "/", 17: "(", 18: ")"}

def fit_classifier(args, net, train_set, val_set, device, optimizer, criterion, is_train=True):
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
        Loss, accuracy, error_img, error_pred = [], [], [], []
        start_time = time.time()
        for batch_idx, data in enumerate(dataset):
            # Get data ops
            img_batch, label_batch = data[0][0].to(device), data[0][1].to(device)
            #subs = [invert_label[int(label_batch[i])] for i in range(label_batch.size(0))]
            #vb.plot_tensor(img_batch, sub_title=subs, path=TMPJPG, deNormalize=True, ratio=3, bg_color=200)
            pred = net(img_batch)#, img_batch[:,:,:,:8])
            _, pred_idx = torch.max(pred, dim=1)
            correct = [1 if int(pred_idx[i]) == int(label_batch[i]) else 0 for i in range(pred_idx.size(0))]
            if not is_train:
                error_img += [img_batch[i, :, :, :].unsqueeze(0) for i, _ in enumerate(correct) if _ is 0]
                error_pred += [
                    "%2s" % (invert_label[int(label_batch[i])]) + "=>" + "%2s" % (invert_label[int(pred_idx[i])])
                    for i, _ in enumerate(correct) if _ is 0]
            accuracy.append(sum(correct) / pred.size(0))
            loss = criterion(pred, label_batch)
            Loss.append(float(loss.data))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(prefix + " --- loss: %8f, accuracy: %4s, at epoch %04d/%04d, cost %3f seconds ---" %
              (sum(Loss) / len(Loss), str(sum(accuracy) / len(accuracy))[:6],
               epoch + 1, args.epoches_per_phase, time.time() - start_time))
        epoch_Loss += Loss
        if is_train:
            args.curr_epoch += 1
        else:
            if error_img and args.curr_epoch != 0:
                path = os.path.expanduser("~/Pictures/tmp_error_pred_%s.jpg" % (str(args.curr_epoch).zfill(4)))
                vb.plot_tensor(torch.cat(error_img, dim=0), path=path, title="Error Prediction", sub_title=error_pred,
                               ratio=1.6666, deNormalize=True, font_size=1)
            else:
                print("Everything is fine")
    if is_train:
        # start = time.time()
        # vb.visualize_gradient(args, net, minimun_rank=4)
        # print("Gradient visualization completed, took %3f second" % (time.time() - start))
        vb.plot_loss_distribution([epoch_Loss], keyname=None, save_path=os.path.expanduser(args.loss_log),
                                  name="Loss_trend", epoch=args.curr_epoch, weight=None)
        return net

def verify_args(args, presets, options=None):
    args = util.prepare_args(args, presets, options=options)
    # Verification Code Here
    if args.batch_size == 1:
        args.batch_norm = False
    return args


def main():
    cls_args = BaseOptions().initialize()
    seg_args = BaseOptions().initialize()
    cls_args = verify_args(cls_args, preset.PRESET, options=["ocr1", "U_01", "runtime"])
    seg_args = verify_args(seg_args, seg_preset.PRESET, options=["ocr1", "U_01", "runtime"])
    if cls_args.deterministic_train:
        torch.manual_seed(cls_args.seed)
    device = torch.device("cuda:" + cls_args.gpu_id)
    #device = torch.device("cpu")
    
    # Prepare Dataset
    open_corpus = seg_misc.fetch_data(seg_args, sources=["open_cor"], batch_size=1)
    temp_holding = seg_misc.fetch_data(seg_args, sources=["test_set"], batch_size=1, txt_file=["numbers.txt"])
    train_set, val_set = misc.fetch_classification_data(cls_args, sources=cls_args.img_folder_name,
                                                        foldername=cls_args.img_folder_name,
                                                        batch_size=cls_args.batch_size)
    
    # Prepare Network
    seg_net = seg_model.OCR_Segmenter(bottleneck=seg_args.bottleneck_size, BN=seg_args.batch_norm,
                                      output_size=seg_args.model_load_size[-1], dropout=0).to(device)
    seg_net = util.load_latest_model(seg_args, seg_net)
    cls_net = model.Classifier(bottleneck=cls_args.bottleneck_size, BN=cls_args.batch_norm,
                               output_size=cls_args.output_size, device=device).to(device)
    #test.seperate(seg_args, cls_args, seg_net, cls_net, test_set, device, 0, plot_result=True)

    if cls_args.finetune:
        cls_net = util.load_latest_model(cls_args, cls_net)
        test.test(seg_args, cls_args, seg_net, cls_net, open_corpus, device, 0, plot_result=False)
        test.test(seg_args, cls_args, seg_net, cls_net, temp_holding, device, 0, plot_result=False)
    
    # Prepare loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls_net.parameters(), lr=cls_args.learning_rate, weight_decay=cls_args.weight_decay)
    
    # Train, Val and Test
    open_record = 0.0
    tmp_record = 0.0
    for epoch in range(cls_args.epoch_num):
        print("\n ======================== PHASE: %s/%s ========================= " %
              (str(int(cls_args.curr_epoch / cls_args.epoches_per_phase)).zfill(4), str(cls_args.epoch_num).zfill(4)))
        fit_classifier(cls_args, cls_net, train_set, val_set, device, optimizer, criterion, is_train=False)
        fit_classifier(cls_args, cls_net, train_set, val_set, device, optimizer, criterion, is_train=True)
        #util.save_model(cls_args, cls_args.curr_epoch, cls_net.state_dict())
        if epoch >= 10:
            if epoch % 2 == 0:
                print("Testing on Open Corpus")
                open_record = test.test(seg_args, cls_args, seg_net, cls_net, open_corpus, device, open_record,
                                        plot_result=False, prefix="open")
            else:
                print("Testing on Tempholdings")
                tmp_record = test.test(seg_args, cls_args, seg_net, cls_net, temp_holding, device, tmp_record,
                                       plot_result=False, prefix="temp")

if __name__ == "__main__":
    main()