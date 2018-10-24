import os, glob, csv
import data.misc as misc
import imgaug
from imgaug import augmenters
import numpy as np
import cv2
import torchvision.transforms as T

def read_image(args, path, seed):
    assert type(seed) is int, "random seed should be int."
    if args.img_channel is 1:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    elif args.img_channel is 3:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("image channel shall be only 1 or 3")
    image = cv2.resize(image, (args.img_size, args.img_size))
    if args.do_affine:
        imgaug.seed(seed)
        image = transform(args).augment_image(image)
    img = T.ToTensor(image)
    if args.img_channel is 1:
        img = T.Normalize((0.5), (0.5))
    elif args.img_channel is 3:
        img = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return image

def transform(args):
    seq = augmenters.Sequential([
        augmenters.Sometimes(0.2, augmenters.AverageBlur(k=5)),
        #augmenters.Sometimes(0.2, augmenters.BilateralBlur(d=2, sigma_color=(10, 250), sigma_space=(10, 250))),
        augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))),
        augmenters.Affine(
            scale={"x": args.scale, "y": args.scale},
            translate_percent={"x": args.translation, "y": args.translation},
            rotate=args.rotation,
            shear=args.shear
        ),
        augmenters.ContrastNormalization((0.75, 1.5)),
        augmenters.Multiply((0.8, 1.2), per_channel=0.2)
    ], random_order=True)
    return seq

def load_path_from_csv(args, len, paths, dig_level=0):
    if type(paths) is str:
        paths = [paths]
    for path in paths:
        with open(path, "r") as csv_file:
            pass

def load_path_from_folder(args, len, paths, dig_level=0):
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
    if self.extensions:
        output = [_ for _ in output if misc.extension_check(_, self.extensions)]
    # 1->A, 2->B, 3->C, ..., 26->Z
    key = misc.number_to_char(len)
    return {key: output}