import os, time
import torch
import numpy as np
from data.arbitrary import Arbitrary
from data.img2img import Img2Img
import data.data_loader as loader
import data.misc as misc
from options.base_options import BaseOptions


def test_Arbitrary(args):
    data = Arbitrary(args=args, root=args.path, sources=["trainA", "trainB", "testA", "testB"],
                     modes=["path"] * 4, load_funcs=[loader.read_image] * 4, dig_level=[0] * 4)
    data.prepare()
    for i in range(len(data)):
        A, B, C, D = data[i]
        pass
    
def test_img2img(args):
    args.path = "~/Pictures/dataset/buddha"
    data = Img2Img(args=args, sources=["trainA", "trainB"], modes=["path"] * 2,
                   load_funcs=[loader.read_image] * 2, dig_level=[0] * 2)
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


def test_super_reso(args):
    args.path = "~/Pictures/dataset/buddha"
    options = {"sizes":[(400, 300), (1024, 768)]}
    data = Img2Img(args=args, sources=["trainA", "trainB"], modes=["path"] * 2,
                   load_funcs=["image"] * 2, dig_level=[0] * 2, **options)
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
    After you extract the downloaded file, the structure will be:
        |
        |-batches.meta
        |-data_batch_1 (training data writen in pickle format)
        |-data_batch_2 (training data writen in pickle format)
        |-data_batch_3 (training data writen in pickle format)
        |-data_batch_4 (training data writen in pickle format)
        |-readme.html
        |-test_batch (test data writen in pickle format)
    """
    import pickle
    args.path = "~/Downloads/cifar-10"
    def load_pickle_data(args, _len, names, dig_level=0):
        data = []
        label = []
        for name in names:
            with open(name, "rb") as db:
                dict = pickle.load(db, encoding="bytes")
                dim = max([len(dict[_]) for _ in dict.keys()])
                for i in range(dim):
                    data.append(reshape(dict[misc.str2bytes("data")][i, :]))
                    label.append(dict[misc.str2bytes("labels")][i])
        return {misc.number_to_char(0):data, misc.number_to_char(1):label}

    def reshape(img):
        """
        transfer a 1D array to RGB image, which cannot be done using:
        np.reshape(32, 32, 3)
        """
        img_R = img[0:1024].reshape((32, 32))
        img_G = img[1024:2048].reshape((32, 32))
        img_B = img[2048:3072].reshape((32, 32))
        return np.dstack((img_R, img_G, img_B))
    
    data = Arbitrary(args=args, load_funcs=[loader.to_tensor, loader.just_return_it],
                     sources=[("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4")],
                     modes=[load_pickle_data], dig_level=[0])
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


if __name__ == "__main__":
    args = BaseOptions().initialize()
    test_img2img(args)
    test_super_reso(args)
    test_cifar(args)
    #test_Arbitrary(args)