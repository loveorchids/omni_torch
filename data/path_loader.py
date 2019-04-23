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

import glob, pickle
import omni_torch.data.misc as misc
import numpy as np
import omni_torch.data as data


def load_path_from_csv(args, length, paths, dig_level=0):
    if type(paths) is str:
        paths = [paths]
    for path in paths:
        with open(path, "r") as csv_file:
            pass

def load_path(args, path, dig_level):
    current_folders = [path]
    # Do not delete the following line, we need this when dig_level is 0.
    sub_folders = []
    while dig_level > 0:
        sub_folders = []
        for sub_path in current_folders:
            sub_folders += glob.glob(sub_path + "/*")
        current_folders = sub_folders
        dig_level -= 1
    sub_folders = []
    for _ in current_folders:
        sub_folders += glob.glob(_ + "/*")
    if args.extensions:
        sub_folders = [_ for _ in sub_folders if misc.extension_check(_, args.extensions)]
    return sub_folders

def load_path_from_folder(args, length, paths, dig_level=0):
    """
    'paths' is a list or tuple, which means you want all the sub paths within 'dig_level' levels.
    'dig_level' represent how deep you want to get paths from.
    """
    output = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:
        sub_folders = load_path(args, path, dig_level)
        output += sub_folders
    output.sort()
    return [output]

def load_path_from_multi_folder(args, length, paths, dig_level=0):
    """
    The only difference from its sibling "load_path_from_folder"
    is that the sibling function merge all sub_folders into one list,
    while this function load them separately.
    :param args:
    :param length:
    :param paths: a list or tuple, which means you want all the sub paths within 'dig_level' levels.
    :param dig_level: represent how deep you want to get paths from.
    :return:
                [[source_1_0, source_2_0, ..., source_n_0],
                 [source_1_1, source_2_1, ..., source_n_1],
                ... ,
                 [source_1_m, source_2_m, ..., source_n_m]]
    """
    output = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:
        sub_folders = load_path(args, path, dig_level)
        sub_folders.sort()
        output.append(sub_folders)
    # Make sure all the sub_folders contains same number of path
    assert max([len(_) for _ in output]) == min([len(_) for _ in output])
    return [list(zip(*output))]

def load_cifar_from_pickle(args, length, names, dig_level=0):
    """
    Cifar Dataset Structure
        |
        |-batches.meta
        |-data_batch_1 (training data writen in pickle format)
        |-data_batch_2 (training data writen in pickle format)
        |-data_batch_3 (training data writen in pickle format)
        |-data_batch_4 (training data writen in pickle format)
        |-readme.html
        |-test_batch (test data writen in pickle format)
    """
    def reshape(img):
        """
        transfer a 1D array to RGB image, which cannot be done by numpy reshape
        """
        img_R = img[0:1024].reshape((32, 32))
        img_G = img[1024:2048].reshape((32, 32))
        img_B = img[2048:3072].reshape((32, 32))
        return np.dstack((img_R, img_G, img_B))
    
    data = []
    label = []
    if type(names) is str:
        names = [names]
    for name in names:
        with open(name, "rb") as db:
            dict = pickle.load(db, encoding="bytes")
            dim = max([len(dict[_]) for _ in dict.keys()])
            for i in range(dim):
                data.append(reshape(dict[misc.str2bytes("data")][i, :]))
                label.append(dict[misc.str2bytes("labels")][i])
    return data, label

def load_img_from_path(args, length, names, dig_level=0):
    """
    Not recommanded to use unless you have a small scale dataset
    :param args:
    :param length:
    :param names:
    :param dig_level:
    :return:
    """
    import cv2
    import data.data_loader_ops as dop
    data = []
    paths = load_path_from_folder(args, length, names, dig_level)[0]
    while len(paths) is not 0:
        cell = paths.pop()
        if args.img_channel is 1:
            image = cv2.imread(cell, 0)
        else:
            image = cv2.cvtColor(cv2.imread(cell), cv2.COLOR_BGR2RGB)
        data += dop.segment_image(image, args, None, None, None, False)
    return [data]