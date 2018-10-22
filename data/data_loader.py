import os, glob
import data.misc as misc
import imgaug
from imgaug import augmenters
import numpy as np
import cv2

def read_image(args, path, seed):
    assert type(seed) is int, "random seed should be int."
    if args.img_channel is 1:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    elif args.img_channel is 3:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.img_size, args.img_size))
    imgaug.seed(seed)
    image = transform().augment_image(image)
    return image

def transform():
    seq = augmenters.Sequential([
        augmenters.Sometimes(0.2, augmenters.AverageBlur(k=5)),
        #augmenters.Sometimes(0.2, augmenters.BilateralBlur(d=2, sigma_color=(10, 250), sigma_space=(10, 250))),
        augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))),
        augmenters.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-15, 15),
            shear=(-8, 8)
        ),
        augmenters.ContrastNormalization((0.75, 1.5)),
        augmenters.Multiply((0.8, 1.2), per_channel=0.2)
    ], random_order=True)
    return seq