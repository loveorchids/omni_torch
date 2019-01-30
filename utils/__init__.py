"""
# Copyright (c) 2018 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import os, glob, warnings
import torch
import json


def get_stride_padding_kernel(input, out):
    """
    Calculation based on:
    https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
    :param input:
    :param out:
    :return:
    """
    for K in range(1, 6):
        for S in range(1, 6):
            for P in range(6):
                if float(P) == (input - 1 - S * (out - 1)) / 2:
                    return S, P, K


def prepare_args(args, presets, options=None):
    def load_preset(args, preset, preset_code):
        class Bunch:
            def __init__(self, adict):
                self.__dict__.update(adict)
        
        if preset_code.endswith(".json"):
            assert os.path.exists(preset_code)
            path = os.path.expanduser(preset_code)
            with open(path, "r") as file:
                data = json.load(file)
            data = vars(args).update(data)
            return Bunch(data)
        else:
            args = preset[preset_code](args)
            # pattern_dict = pattern.create_args(self.args)
            return args
    
    def save_args(args):
        path = os.path.expanduser(os.path.join(args.path, args.code_name, "preset.json"))
        with open(path, "w") as file:
            json.dump(vars(args), file)
        return
    
    def make_folder(args, name=""):
        path = os.path.join(args.path, args.code_name, name)
        if not os.path.exists(path):
            os.mkdir(path)
        elif not args.cover_exist:
            raise FileExistsError("such code name already exists")
        return path
    
    # Load general and unique options
    if options:
        for option in options:
            args = load_preset(args, presets, option)
    else:
        args = load_preset(args, presets, args.general_options)
        args = load_preset(args, presets, args.unique_options)
        args = load_preset(args, presets, args.runtime_options)
    
    args.path = os.path.expanduser(args.path)
    if args.create_path:
        args.model_dir = make_folder(args)
        args.log_dir = make_folder(args, "log")
        args.loss_log = make_folder(args, "loss")
        args.grad_log = make_folder(args, "grad")
        args.val_log = make_folder(args, "val")
    save_args(args)
    return args


def save_model(args, epoch, state_dict, keep_latest=5):
    model_list = [_ for _ in glob.glob(args.model_dir + "/*.pth") if os.path.isfile(_)]
    model_list.sort()
    if len(model_list) < keep_latest:
        pass
    else:
        remove_list = model_list[: (keep_latest - 1) * -1]
        for item in remove_list:
            os.remove(item)
    model_path = os.path.join(args.model_dir, "epoch_" + str(epoch).zfill(4) + ".pth")
    torch.save(state_dict, model_path)


def load_latest_model(args, net):
    model_list = [_ for _ in glob.glob(args.model_dir + "/*.pth") if os.path.isfile(_)]
    if not model_list:
        warnings.warn("Cannot find models")
        return net
    model_list.sort()
    epoch = int(model_list[-1][model_list[-1].rfind("_") + 1:model_list[-1].rfind(".")])
    print("Load model form: " + model_list[-1])
    if not torch.cuda.is_available():
        model_data = torch.load(model_list[-1], map_location='cpu')
    else:
        model_data = torch.load(model_list[-1])
    try:
        net.load_state_dict(model_data)
    except RuntimeError:
        warnings.warn("Model shape does not matches!")
        return net
    args.curr_epoch = epoch
    return net


def normalize_image(args, img, mean=None, std=None, bias=None):
    if args is None:
        img_mean, img_std, img_bias = mean, std, bias
    else:
        img_mean, img_std, img_bias = args.img_mean, args.img_std, args.img_bias
    if img.shape[2] == 1:
        mean = 0.29 * img_mean[0] + 0.59 * img_mean[1] + 0.12 * img_mean[2]
        std = sum(img_std) / len(img_std)
        bias = sum(img_bias) / len(img_bias)
    elif img.shape[2] == 3:
        mean, std, bias = img_mean, img_std, img_bias
    else:
        raise RuntimeError("image channel should either be 1 or 3")
    return (img / 255 - mean) / std + bias


def denormalize_image(args, img, mean=None, std=None, bias=None):
    if args is None:
        img_mean, img_std, img_bias = mean, std, bias
    else:
        img_mean, img_std, img_bias = args.img_mean, args.img_std, args.img_bias
    if img.shape[2] == 1:
        mean = 0.29 * img_mean[0] + 0.59 * img_mean[1] + 0.12 * img_mean[2]
        std = sum(img_std) / len(img_std)
        bias = sum(img_bias) / len(img_bias)
    elif img.shape[2] == 3:
        mean, std, bias = img_mean, img_std, img_bias
    else:
        raise RuntimeError("image channel should either be 1 or 3")
    return 255 * ((img - bias) * std + mean)


if __name__ == "__main__":
    pass
    # save_model(1, 2, 3, os.path.expanduser("~/Documents/tmp"))
