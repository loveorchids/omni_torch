import os, time
import multiprocessing as mpi
import torch
from data.arbitrary import Arbitrary
from data.img2img import Img2Img
import data.mode as mode
import data.data_loader as loader
import data.data_loader_ops as dop
from options.base_options import BaseOptions

    
def test_img2img(args):
    args.path = "~/Pictures/dataset/buddha"
    args.batch_size = 4
    # Uncomment if you want load images in random order
    # args.random_order_load = True
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

def test_img2img_advanced(args):
    args.path = "~/Pictures/dataset/buddha"
    args.random_crop = False
    args.segments = [4, 4]
    # Uncomment if you want load images in random order
    # args.random_order_load = True
    options = {"sizes": [(1200, 1200), (1920, 1920)]}
    args.batch_size = 1
    data = Img2Img(args=args, sources=["trainA", "trainB"], modes=["path"] * 2,
                   load_funcs=[loader.read_image] * 2, dig_level=[0] * 2,
                   loader_ops=[dop.segment_image] * 2, **options)
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
    """
    args.path = "~/Downloads/cifar-10"
    args.batch_size = 128
    data = Arbitrary(args=args, load_funcs=[loader.to_tensor, loader.just_return_it],
                     sources=[("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4")],
                     modes=[mode.load_cifar_from_pickle], dig_level=[0])
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

def test_multilayer_input(args):
    import cv2
    args.path = "~/Pictures/dataset/buddha"
    args.batch_size = 4
    args.img_channels = 1
    # Uncomment if you want load images in random order
    # args.random_order_load = True
    def combine_multi_image(args, paths, seed, size, ops):
        imgs = []
        for path in paths:
            imgs.append(loader.read_image(args, path, seed, size, ops))
        return torch.cat(imgs)
    data = Arbitrary(args=args, sources=[("groupa", "groupb"), "groupb"], modes=["sep_path", "path"],
                     load_funcs=[combine_multi_image, loader.read_image],
                     dig_level=[0] * 2, **{"ops": cv2.bitwise_not})
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
    test_cifar(args)
    test_img2img(args)
    test_img2img_advanced(args)
    test_super_reso(args)
    test_multilayer_input(args)