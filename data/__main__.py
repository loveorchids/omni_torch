import os
from data.arbitrary import Arbitrary
from data.img2img import Img2Img
import data.data_loader as loader

from options.base_options import BaseOptions

def read_image(args, path, seed):
    import cv2, imgaug
    import torchvision.transforms as T
    assert type(seed) is int, "random seed should be int."
    if args.img_channel is 1:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    elif args.img_channel is 3:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("image channel shall be only 1 or 3")
    image = cv2.resize(image, (args.img_size_2, args.img_size_2))
    if args.do_affine:
        imgaug.seed(seed)
        image = loader.transform(args).augment_image(image)
    img = T.ToTensor(image)
    if args.img_channel is 1:
        img = T.Normalize(0.5, 0.5)
    elif args.img_channel is 3:
        img = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return image

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

def test_super_res(args):
    data = Img2Img(args=args, root=args.path, sources=["trainA", "trainA"],
                   modes=["path"] * 2, load_funcs=[loader.read_image, read_image], dig_level=[0] * 2,
                   extensions=["jpg", "jpeg", "JPG", "PNG", "png"])
    
if __name__ == "__main__":
    args = BaseOptions().initialize()
    args.path = "~/Pictures/dataset/buddha"
    args.img_size_2 = 1024
    test_img2img(args)
    #test_Arbitrary(args)