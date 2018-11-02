import glob, pickle
import data.misc as misc
import numpy as np


def load_path_from_csv(args, length, paths, dig_level=0):
    if type(paths) is str:
        paths = [paths]
    for path in paths:
        with open(path, "r") as csv_file:
            pass

def load_path_from_folder(args, length, paths, dig_level=0):
    """
    'paths' is a list or tuple, which means you want all the sub paths within 'dig_level' levels.
    'dig_level' represent how deep you want to get paths from.
    """
    output = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:
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
        output += sub_folders
    if args.extensions:
        output = [_ for _ in output if misc.extension_check(_, args.extensions)]
    return [output]

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
