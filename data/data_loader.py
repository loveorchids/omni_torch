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

