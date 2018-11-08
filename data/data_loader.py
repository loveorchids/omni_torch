import imgaug, cv2
from imgaug import augmenters
import torch
import torchvision.transforms as T
import numpy as np
import data.misc as misc
import data

ALLOW_WARNING = data.ALLOW_WARNING

def read_image(args, path, seed, size, ops=None):
    if args.img_channel is 1:
        image = cv2.imread(path, 0)
    else:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    image = misc.random_crop(image, args.load_size, seed)
    #print("before: " + str(image.shape))
    if args.do_imgaug:
        imgaug.seed(seed)
        image = prepare_augmentation(args).augment_image(image)
    if args.do_resize:
        size = (size[0], size[1])
    else:
        size = (image.shape[0], image.shape[1])
    ver = 8 * round(size[0] / 8)
    height = 8 * round(size[1] / 8)
    image = cv2.resize(image, (ver, height))
    if ops:
        image = ops(image)
    if args.img_channel is 1:
        image = np.expand_dims(image, axis=-1)
    #print("after: " + str(image.shape))
    return to_tensor(args, image, seed, size, ops)

def to_tensor(args, image, seed, size, ops=None):
    trans = T.Compose([T.ToTensor(), T.Normalize(args.img_mean, args.img_std)])
    return trans(image)

def to_tensor_with_aug(args, image, seed, size, ops=None):
    if args.do_imgaug:
        imgaug.seed(seed)
        image = prepare_augmentation(args).augment_image(image)
    return to_tensor(args, image, seed, size, ops)

def just_return_it(args, data, seed=None, size=None, ops=None):
    """
    Because the label in cifar dataset is int
    So here it will be transfered to a torch tensor
    """
    return torch.tensor(data)

def prepare_augmentation(args):
    aug_list = []
    if args.affine_trans:
        aug_list.append(augmenters.Affine(scale={"x": args.scale, "y": args.scale},
                                      translate_percent={"x": args.translation, "y": args.translation},
                                      rotate=args.rotation, shear=args.shear, cval=args.aug_bg_color))
    if args.random_crop:
        aug_list.append(augmenters.Crop(percent=args.crop_percent, keep_size=True))
    if args.random_flip:
        aug_list.append(augmenters.Fliplr(0.5))
        aug_list.append(augmenters.Flipud(0.0))
    if args.random_brightness:
        aug_list.append(augmenters.ContrastNormalization((0.75, 1.5)))
        aug_list.append(augmenters.Multiply((0.9, 1.1), per_channel=0.2))
    if args.random_noise:
        aug_list.append(augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))))
        aug_list.append(augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                         per_channel=0.5))
    seq = augmenters.Sequential(aug_list, random_order=True)
    return seq

def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]
