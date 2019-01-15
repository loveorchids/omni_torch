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
# Classifier
import networks.ocr.classifier.models as cls_model
import networks.ocr.classifier.presets as cls_preset

TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
invert_label_ori = {12: '-', 3: '¥', 11: '0', 2: ',', 7: '〒', 6: '3', 5: '4', 8: '7', 14: '5', 0: '.', 9: '1',
                13: '8', 4: '6', 1: '9', 10: '2', 15: '', 16: "/", 17:"(", 18: ")"}
# Treat dot as comma, see key 0 and 2
_invert_label = {12: '-', 3: '¥', 11: '0', 2: ',', 7: '〒', 6: '3', 5: '4', 8: '7', 14: '5', 0: '.', 9: '1',
                13: '8', 4: '6', 1: '9', 10: '2', 15: '', 16: "/", 17:"(", 18: ")"}
_invert_label_no_dot_comma = {12: '-', 3: '¥', 11: '0', 2: '', 7: '〒', 6: '3', 5: '4', 8: '7', 14: '5', 0: '', 9: '1',
                13: '8', 4: '6', 1: '9', 10: '2', 15: '', 16: "/", 17:"(", 18: ")"}
_invert_label_no_sp_character = {12: '', 3: '', 11: '0', 2: '', 7: '〒', 6: '3', 5: '4', 8: '7', 14: '5', 0: '', 9: '1',
                13: '8', 4: '6', 1: '9', 10: '2', 15: '', 16: "", 17:"", 18: ""}

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

def test(seg_args, cls_args, segmenter, classifier, dataset, device, highest_record, plot_result=True, prefix=""):
    os.system("rm /home/wang/Pictures/*%s_tmp_*"%(prefix))
    segmenter.eval()
    classifier.eval()
    with torch.no_grad():
        accuracy = 0
        accuracy_no_dot = 0
        accuracy_no_sp = 0
        for idx, data in enumerate(dataset):
            if idx in[]:
                print("")
            sep = []
            start = 0
            try:
                gt_label = "".join([_invert_label[int(data[0][1][0][i])] for i in range(data[0][1].size(1))])
            except ValueError:
                pass
            except KeyError:
                # means is nothing in the label
                accuracy += 1
                continue
            try:
                gt_label_no_dot = "".join([_invert_label_no_dot_comma[int(data[0][1][0][i])] for i in range(data[0][1].size(1))])
            except KeyError:
                accuracy_no_dot += 1
                continue
            try:
                gt_label_no_sp = "".join([_invert_label_no_sp_character[int(data[0][1][0][i])] for i in range(data[0][1].size(1))])
            except:
                accuracy_no_sp += 1
                continue
            input_img = data[0][0]
            pred, pred_no_dot, pred_no_sp = [], [], []
            while start + 5 < input_img.size(3):
                # while start + args.model_load_size[-1] < img.size(3):
                slice = get_image_slice(input_img, start, seg_args.model_load_size[-1], device)
                seg_pred = segmenter(slice, seg_args.model_load_size)
                _, seg_idx = torch.max(seg_pred, dim=1)
                cls_img = slice[:, :, :, :int(seg_idx)]
                if cls_args.allow_left_aux :
                    if start - cls_args.left_aux_width >= 0:
                        left_aux = input_img[:, :, :, start - cls_args.left_aux_width: start].to(device)
                        cls_img = torch.cat([left_aux, cls_img], dim=3)
                else:
                    cls_args.left_aux_width = 0
                    
                if cls_args.allow_right_aux :
                    if start + int(seg_idx) + 1 + cls_args.right_aux_width <= input_img.size(3):
                        right_aux = input_img[:, :, :, start + int(seg_idx): start + int(seg_idx) + cls_args.left_aux_width].to(device)
                        cls_img = torch.cat([cls_img, right_aux], dim=3)
                else:
                    cls_args.right_aux_width = 0
                    
                if cls_args.stretch_img:
                    cls_img = tf.interpolate(cls_img, size=(cls_args.resize_height, cls_args.resize_height))
                else:
                    component = torch.zeros(slice.size(0), slice.size(1), slice.size(2), cls_args.resize_height - cls_img.size(3)).to(device)
                    cls_img = torch.cat([component, cls_img], dim=3)
                cls = classifier(cls_img)
                _, cls_idx = torch.max(cls, dim=1)
                if int(cls_idx) != 15:
                    pred.append(_invert_label[int(cls_idx)])
                    pred_no_dot.append(_invert_label_no_dot_comma[int(cls_idx)])
                    pred_no_sp.append(_invert_label_no_sp_character[int(cls_idx)])
                if int(seg_idx) == 0:
                    start += (int(seg_idx) + 1)
                else:
                    start += int(seg_idx)
                sep.append(start)
            #pred = calibration(pred)
            pred = ''.join(pred)
            if pred == gt_label:
                accuracy += 1
            if "".join(pred_no_sp) == gt_label_no_sp:
                accuracy_no_sp += 1
            if "".join(pred_no_dot) == gt_label_no_dot:
                accuracy_no_dot += 1
            elif plot_result:
                pred_img = misc.draw_multi_line(input_img, coord=sep)
                path = os.path.expanduser("~/Pictures/%s_tmp_%s.jpg" % (prefix, idx))
                vb.plot_tensor(pred_img, path, deNormalize=True, bg_color=224, margin=10, sub_title=[gt_label + "=>" + pred])
        print("Accuracy is: %s"%(accuracy/len(dataset)))
        print("Accuracy_no_dot is: %s" % (accuracy_no_dot / len(dataset)))
        print("Accuracy_no_sp is: %s" % (accuracy_no_sp / len(dataset)))
        if accuracy/len(dataset) > highest_record:
            highest_record = accuracy/len(dataset)
            util.save_model(cls_args, cls_args.curr_epoch, classifier.state_dict())
            print("Model Saved")
        return highest_record

def seperate(seg_args, cls_args, segmenter, classifier, dataset, device, highest_record, plot_result=True):
    print("Testing...")
    segmenter.eval()
    with torch.no_grad():
        file = open(os.path.expanduser("~/Pictures/open_cor/label.txt"), "w")
        for idx, data in enumerate(dataset):
            if idx % 100 == 0 and idx > 0:
                print("%s samples segmented"%(idx))
            try:
                gt_label = "".join([_invert_label[int(data[0][1][0][i])] for i in range(data[0][1].size(1))])
            except:
                continue
            #gt_label = gt_label.replace("¥", "\\")
            input_img = data[0][0]
            img_path = os.path.join(os.path.expanduser("~/Pictures/open_cor"), str(idx).zfill(4) + ".jpg")
            vb.plot_tensor(input_img, img_path, deNormalize=True, margin=0)
            if not os.path.exists(img_path):
                print(img_path)
            sep = []
            start = 0
            while start< input_img.size(3):
                # while start + args.model_load_size[-1] < img.size(3):
                slice = get_image_slice(input_img, start, seg_args.model_load_size[-1], device)
                seg_pred = segmenter(slice, seg_args.model_load_size)
                _, seg_idx = torch.max(seg_pred, dim=1)
                start += (int(seg_idx) + 1)
                sep.append(str(start))
            file.write(str(idx).zfill(4) + ".jpg" +":" + gt_label + ":" + ",".join(sep) + "\n")
        file.close()

def calibration(prediction):
    if prediction[0] == '¥':
        dot_index = [i for i in range(len(prediction)) if prediction[i] in ["."]]
        if prediction[-1] in [",", "/", "(", "-", ")"]:
            # if the last character is ")", "(", "/", "." then remove it
            prediction.pop()
        elif dot_index[-1] + 3 >= len(prediction):
            prediction[dot_index[-1]] = "."
        elif len(dot_index) == 0:
            pos = len(prediction)
            while pos - 4 > 0:
                pos -= 3
                prediction.insert(pos, ",")
        return prediction
    elif prediction[0] == "〒":
        # Remove all the commas in the postcode
        return [pred for pred in prediction if pred not in [",", ")", "(", "/", "¥"]]
    elif "," in prediction:
        if prediction[-1] == ")":
            prediction.pop()
            dot_index = [i for i in range(len(prediction)) if prediction[i] in ["."]]

            prediction.append(")")
        elif prediction[-1] in [",", "/", "(", "-", ")"]:
            prediction.pop()
            dot_index = [i for i in range(len(prediction)) if prediction[i] in ["."]]

def verify_args(args, presets, options=None):
    args = util.prepare_args(args, presets, options)

    args.loss_weight = dict(zip(args.loss_name, args.loss_weight))
    args.loss_weight_range = dict(zip(args.loss_name, args.loss_weight_range))

    assert args.batch_size == 1, "set batch size to 1 makes the training result better"
    assert args.loss_name[0] == "p_mse"
    assert 0.9999 <= sum([_ for _ in args.loss_weight.values()]) <= 1.0001
    assert all([True if _[0] >= 0 and _[1] <= 1 else False for _ in args.loss_weight_range.values()])
    return args

if __name__ == "__main__":
    seg_args = BaseOptions().initialize()
    cls_args = BaseOptions().initialize()
    seg_args = verify_args(seg_args, preset.PRESET, options=["ocr1", "U_01", "runtime"])
    cls_args = verify_args(cls_args, cls_preset.PRESET, options=["ocr1", "U_01", "runtime"])

    if seg_args.deterministic_train:
        torch.manual_seed(seg_args.seed)
    device = torch.device("cuda:" + seg_args.gpu_id)

    test_set = misc.fetch_data(seg_args, sources=["open_cor"], batch_size=1)

    seg_net = model.OCR_Segmenter(bottleneck=seg_args.bottleneck_size, output_size=seg_args.model_load_size[-1],
                                  dropout=0, BN=seg_args.batch_norm).to(device)
    cls_net = cls_model.Classifier(bottleneck=cls_args.bottleneck_size, BN=cls_args.batch_norm,
                                   output_size=cls_args.output_size).to(device)

    criterion = [loss.JS_Divergence(), nn.MSELoss()]
    optimizer = torch.optim.Adam(seg_net.parameters(), lr=seg_args.learning_rate, weight_decay=seg_args.weight_decay)

    if seg_args.finetune:
        seg_net = util.load_latest_model(seg_args, seg_net)
        cls_net = util.load_latest_model(cls_args, cls_net)

    test(seg_args, cls_args, seg_net, cls_net, test_set, device)