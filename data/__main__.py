import os
from data.arbitrary import Arbitrary
from data.img2img import Img2Img
import data.data_loader as loader

from options.base_options import BaseOptions

def test_Arbitrary(args):
    data = Arbitrary(args=args, root=args.path, sources=["trainA", "trainB", "testA", "testB"],
                     modes=["path"] * 4, load_funcs=[loader.read_image] * 4, dig_level=[0] * 4,
                     extensions=["jpg", "jpeg", "JPG", "PNG", "png"])
    data.prepare()
    for i in range(len(data)):
        A, B, C, D = data[i]
        pass
    
def test_img2img(args):
    data = Img2Img(args=args, root=args.path, sources=["trainA", "trainB"],
                     modes=["path"] * 2, load_funcs=[loader.read_image] * 2, dig_level=[0] * 2,
                     extensions=["jpg", "jpeg", "JPG", "PNG", "png"])
    data.prepare()
    for i in range(len(data)):
        A, B, C, D = data[i]
        pass
    
if __name__ == "__main__":
    args = BaseOptions().initialize()
    args.path = "~/Pictures/dataset/buddha"
    test_img2img(args)
    #test_Arbitrary(args)