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

import os, time
import multiprocessing as mpi
import torch
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
from omni_torch.data.img2img_dataset import Img2Img_Dataset
import omni_torch.data.path_loader as mode
import omni_torch.data.data_loader as loader
import omni_torch.data.data_loader_ops as dop
from omni_torch.options.base_options import BaseOptions

    
def test_img2img(args):
    args.path = "~/Pictures/dataset/buddha"
    args.batch_size = 4
    # Uncomment if you want load images in random order
    # args.random_order_load = True
    data = Img2Img_Dataset(args=args, sources=["trainA", "trainB"], step_1=["path"] * 2,
                           step_2=[loader.read_image] * 2, auxiliary_info=[0] * 2)
    data.prepare()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(data,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(device), target.to(device)
        print("--- %s seconds to read data batch %d---" % (time.time() - start_time, batch_idx))
        print(data.shape)
        print(target.shape)

def test_img2img_advanced(args):
    args.path = "~/Pictures/dataset/buddha"
    args.do_crop_to_fix_size = False
    args.segments = [4, 4]
    # Uncomment if you want load images in random order
    # args.random_order_load = True
    options = {"sizes": [(1200, 1200), (1920, 1920)]}
    args.batch_size = 1
    data = Img2Img_Dataset(args=args, sources=["trainA", "trainB"], step_1=["path"] * 2,
                           step_2=[loader.read_image] * 2, auxiliary_info=[0] * 2,
                           pre_process=[dop.segment_image] * 2, **options)
    data.prepare()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(device), target.to(device)
        print("--- %s seconds to read data batch %d---" % (time.time() - start_time, batch_idx))
        print(data.shape)
        print(target.shape)


def test_super_reso(args):
    args.path = "~/Pictures/dataset/buddha"
    # Uncomment if you want load images in random order
    # args.random_order_load = True
    options = {"sizes":[(400, 300), (1024, 768)]}
    args.batch_size = 2
    data = Img2Img_Dataset(args=args, sources=["trainA", "trainB"], step_1=["path"] * 2,
                           step_2=["image"] * 2, auxiliary_info=[0] * 2, **options)
    data.prepare()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(device), target.to(device)
        print("--- %s seconds to read data batch %d---" % (time.time() - start_time, batch_idx))
        print(data.shape)
        print(target.shape)


def test_cifar(args):
    """
    You can download cifar at: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """
    args.path = "~/Downloads/cifar-10"
    args.batch_size = 128
    data = Arbitrary_Dataset(args=args, step_2=[loader.to_tensor, loader.just_return_it],
                             sources=[("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4")],
                             step_1=[mode.load_cifar_from_pickle], auxiliary_info=[0])
    data.prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(device), target.to(device)
        print("--- %s seconds to read data batch %d---" % (time.time() - start_time, batch_idx))
        print(data.shape)
        print(target.shape)
        
def test_segmented_input(args):
    args.path = "~/Pictures/dataset/buddha"
    args.batch_size = 12
    args.img_channels = 1
    args.do_imgaug = False
    data = Arbitrary_Dataset(args=args, sources=["trainA", "trainB"], step_1=[mode.load_img_from_path] * 2,
                             step_2=[loader.to_tensor] * 2, auxiliary_info=[0] * 2)
    data.prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    for batch_idx, (data, target) in enumerate(data_loader):
        start_time = time.time()
        data, target = data.to(device), target.to(device)
        print("--- %s seconds to read data batch %d---" % (time.time() - start_time, batch_idx))
        print(data.shape)
        print(target.shape)
    

if __name__ == "__main__":
    args = BaseOptions().initialize()
    test_segmented_input(args)
    #test_cifar(args)
    #test_img2img(args)
    test_img2img_advanced(args)
    test_super_reso(args)
    #test_multilayer_input(args)