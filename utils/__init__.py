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

import os, glob, warnings, datetime, itertools
import torch
import numpy as np
import random
import json
from omni_torch.options.base_options import BaseOptions
import omni_torch.options.options_edict as edict_options


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

def get_args(preset):
    settings = BaseOptions().initialize()
    args = edict_options.initialize()
    args = prepare_args(args, preset, options=settings.which)
    if args.deterministic_train:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    if args.gpu_id == "cpu":
        warnings.warn("You are using CPU in training")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:" + args.gpu_id)
    return args

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
    return args


def save_args(args):
    path = os.path.expanduser(os.path.join(args.path, args.code_name, "preset_%s.json"%(datetime.datetime.now())))
    with open(path, "w") as file:
        json.dump(vars(args), file)
    return


def save_model(args, epoch, state_dict, keep_latest=5, prefix=None):
    def _save_model(_args, _epoch, _state_dict, _keep_latest=5, _predix=None):
        if _predix:
            model_list = [_ for _ in glob.glob(_args.model_dir + "/%s_*.pth"%(_predix)) if os.path.isfile(_)]
            model_path = os.path.join(_args.model_dir, "%s_epoch_%s.pth" % (_predix, str(_epoch).zfill(4)))
        else:
            model_list = [_ for _ in glob.glob(_args.model_dir + "/*.pth") if os.path.isfile(_)]
            model_path = os.path.join(_args.model_dir, "epoch_%s.pth" % (str(_epoch).zfill(4)))
        model_list.sort()
        if len(model_list) < _keep_latest:
            pass
        else:
            remove_list = model_list[: (_keep_latest - 1) * -1]
            for item in remove_list:
                os.remove(item)
        torch.save(_state_dict, model_path)
        print("Model saved to: %s"%(model_path))
    if type(state_dict) is list or type(state_dict) is tuple:
        assert prefix is not None, "you must specify the prefix of each model"
        assert type(prefix) is list or type(prefix) is tuple, "input param 'prefix' should either be a list or a tuple"
        assert len(state_dict) == len(prefix), "input param 'state_dict' contains %s network while prefix has %s element"\
                                        %(len(state_dict), len(prefix))
        for i, dict in enumerate(state_dict):
            _save_model(args, epoch, dict, keep_latest, prefix[i])
    else:
        _save_model(args, epoch, state_dict, keep_latest, prefix)


def load_latest_model(args, net, prefix=None, return_state_dict=False, nth=1):
    def _load_latset_model(_args, _net, _prefix, return_state_dict, _nth):
        if _prefix:
            model_list = [_ for _ in glob.glob(_args.model_dir + "/%s_*.pth" % (_prefix)) if os.path.isfile(_)]
        else:
            model_list = [_ for _ in glob.glob(_args.model_dir + "/*.pth") if os.path.isfile(_)]
        if not model_list:
            warnings.warn("Cannot find models")
            return _net
        model_list.sort()
        _nth = -1 * _nth
        epoch = int(model_list[_nth][model_list[_nth].rfind("_") + 1:model_list[_nth].rfind(".")])
        print("Load model form: " + model_list[_nth])
        if not torch.cuda.is_available():
            model_data = torch.load(model_list[_nth], map_location='cpu')
        else:
            model_data = torch.load(model_list[_nth])
        if return_state_dict:
            return model_data
        try:
            _net.load_state_dict(model_data)
        except RuntimeError:
            warnings.warn("Model shape does not matches!")
            return _net
        _args.curr_epoch = epoch
        return _net
    if type(net) is list or type(net) is tuple:
        assert prefix is not None, "you must specify the prefix of each model"
        assert type(prefix) is list or type(prefix) is tuple, "input param 'prefix' should either be a list or a tuple"
        assert len(net) == len(prefix), "input param 'net' contains %s network while prefix has %s element"\
                                        %(len(net), len(prefix))
        nets = [_load_latset_model(args, n, prefix[i], return_state_dict, nth) for i, n in enumerate(net)]
        return nets
    else:
        return _load_latset_model(args, net, prefix, return_state_dict, nth)


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

def create_chunks(xs, k_fold):
    if k_fold > 1:
        print("input parameter 'split_val' will be automatically set to %s as k_fold is: %s"
              % (1 / k_fold, k_fold))
    n = k_fold
    xs = list(xs)
    random.shuffle(xs)
    ylen = len(xs)
    size = int(ylen / n)
    chnks = [xs[0 + size * i: size * (i + 1)] for i in range(n)]
    leftover = ylen - size * n
    edge = size * n
    for i in range(leftover):
        chnks[i % n].append(xs[edge + i])
    return chnks

def k_fold_cross_validation(args, dataset, batch_size, batch_size_val, k_fold, collate_fn=None):
    from torch.utils.data import sampler, DataLoader, ConcatDataset
    samples = sum([len(d) for d in dataset]) - 1
    kwargs = {'num_workers': args.loading_threads, 'pin_memory': True} if torch.cuda.is_available() else {}
    chunks = create_chunks(range(samples), k_fold)
    train_sets, val_sets = [], []
    for idx in range(k_fold):
        # Due to cross validation will randomly shuffle the data
        # Thus shuffle will be automatically turned of here
        shuffle = False
        train_index = [chnk for i, chnk in enumerate(chunks) if i != idx]
        train_index = list(itertools.chain.from_iterable(train_index))
        val_index = chunks[idx]
        train_sampler = sampler.SubsetRandomSampler(train_index)
        validation_sampler = sampler.SubsetRandomSampler(val_index)
        train_sets.append(DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                     shuffle=shuffle, sampler=train_sampler, collate_fn=collate_fn,**kwargs))
        val_sets.append(DataLoader(ConcatDataset(dataset), batch_size=batch_size_val,
                                   shuffle=shuffle, sampler=validation_sampler, collate_fn=collate_fn,
                                   **kwargs))
    return list(zip(train_sets, val_sets))

def split_train_val_dataset(args, dataset, batch_size,  batch_size_val, split_val, collate_fn=None):
    from torch.utils.data import sampler, DataLoader, ConcatDataset
    samples = sum([len(d) for d in dataset]) - 1
    kwargs = {'num_workers': args.loading_threads, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_index = random.sample(range(samples), samples - int(samples * split_val))
    train_index_set = set(train_index)
    val_index = [i for i in range(samples) if i not in train_index_set]
    train_sampler = sampler.SubsetRandomSampler(train_index)
    val_sampler = sampler.SubsetRandomSampler(val_index)
    train_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                           shuffle=False, sampler=train_sampler, collate_fn=collate_fn,
                           **kwargs)
    val_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size_val,
                         shuffle=False, sampler=val_sampler, collate_fn=collate_fn,
                         **kwargs)
    return [(train_set, val_set)]

if __name__ == "__main__":
    pass
    # save_model(1, 2, 3, os.path.expanduser("~/Documents/tmp"))