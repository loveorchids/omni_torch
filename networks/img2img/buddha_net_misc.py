import torch
import multiprocessing as mpi
from data.set_img2img import Img2Img_Dataset
from data.set_arbitrary import  Arbitrary_Dataset
import data.data_loader as loader
import data.data_loader_ops as dop

def fetch_data(args, sources):
    data = Img2Img_Dataset(args=args, sources=sources, modes=["path"] * 2,
                           load_funcs=[loader.read_image] * 2, dig_level=[0] * 2,
                           loader_ops=[dop.inverse_image] * 2)
    data.prepare()
    works = mpi.cpu_count() - 2
    kwargs = {'num_workers': 0, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return data_loader


def fetch_new_data(args, sources, ):
    def combine_multi_image(args, paths, seed, size, ops):
        imgs = []
        for path in paths:
            imgs.append(loader.read_image(args, path, seed, size, ops))
        return torch.cat(imgs)
    
    data = Arbitrary_Dataset(args=args, sources=sources, modes=["sep_path", "path"],
                             load_funcs=[combine_multi_image, loader.read_image],
                             loader_ops=[dop.inverse_image] * 2, dig_level=[0] * 2)
    data.prepare()
    works = mpi.cpu_count() - 2
    kwargs = {'num_workers': 0, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return data_loader


def fetch_val_data(args, sources):
    args.segments = [5, 5]
    # Uncomment if you want load images in random order
    # args.random_order_load = True
    options = {"sizes": [(1600, 1600), (1600, 1600)]}
    data = Img2Img_Dataset(args=args, sources=sources, modes=["path"] * 2,
                           load_funcs=[loader.read_image] * 2, dig_level=[0] * 2,
                           loader_ops=[dop.segment_image] * 2, **options)
    data.prepare()
    
    works = mpi.cpu_count() - 2
    kwargs = {'num_workers': 0, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    
    return data_loader