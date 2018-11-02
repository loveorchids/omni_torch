import os, time
import torch
from data.arbitrary import Arbitrary
from data.img2img import Img2Img
import data.mode as mode
import data.data_loader as loader
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
    args.batch_size = 4
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
    def just_return_it(args, data, seed=None, size=None):
        """
        Because the label in cifar dataset is int
        So here it will be transfered to a torch tensor
        """
        return torch.tensor(data)
    
    args.path = "~/Downloads/cifar-10"
    args.batch_size = 128
    data = Arbitrary(args=args, load_funcs=[loader.to_tensor, just_return_it],
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


if __name__ == "__main__":
    args = BaseOptions().initialize()
    test_img2img(args)
    test_super_reso(args)
    test_cifar(args)
    #test_Arbitrary(args)