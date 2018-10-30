import imgaug
from imgaug import augmenters
import cv2
import torch
import torchvision.transforms as T
import data.misc as misc

def read_image(args, path, seed=None, size=None):
    if args.img_channel is 1:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    elif args.img_channel is 3:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("image channel shall be only 1 or 3")
    if args.random_crop:
        image = misc.random_crop(image, args.crop_size, seed)
    if not size:
        size = args.img_size
    else:
        assert len(size) is 2
    image = cv2.resize(image, tuple(size))
    if args.do_affine:
        if seed:
            assert type(seed) is int, "random seed should be int."
            imgaug.seed(seed)
        image = transform(args).augment_image(image)
    image = T.ToTensor()(image)
    if args.img_channel is 1:
        image = T.Normalize((0.5), (0.5))(image)
    elif args.img_channel is 3:
        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
    return image


def to_tensor(args, image, seed=None, size=None):
    if args.do_affine:
        if seed:
            assert type(seed) is int, "random seed should be int."
            imgaug.seed(seed)
        image = transform(args).augment_image(image)
    image = T.ToTensor()(image)
    if args.img_channel is 1:
        image = T.Normalize((0.5), (0.5))(image)
    elif args.img_channel is 3:
        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
    return image

def just_return_it(args, data, seed=None, size=None):
    return torch.tensor(data)

def transform(args):
    aug_list = [augmenters.Sometimes(0.2, augmenters.AverageBlur(k=5)),
                augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))),
                augmenters.Affine(scale={"x": args.scale, "y": args.scale},
                                  translate_percent={"x": args.translation, "y": args.translation},
                                  rotate=args.rotation,shear=args.shear),
                augmenters.ContrastNormalization((0.75, 1.5)),
                augmenters.Multiply((0.8, 1.2), per_channel=0.2)]
    seq = augmenters.Sequential(aug_list, random_order=True)
    return seq

def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]
