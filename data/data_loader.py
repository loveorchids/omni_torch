import random
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import data.misc as misc
import data
from PIL import Image
try:
    import imgaug
    from imgaug import augmenters
except:
    pass

ALLOW_WARNING = data.ALLOW_WARNING

def prepare_image(args, image, seed, size):
    if args.do_imgaug:
        if args.imgaug_engine == "cv2":
            imgaug.seed(seed)
            image = prepare_augmentation(args).augment_image(image)
            # image = misc.random_crop(image, args.crop_size, seed)
        elif args.imgaug_engine == "PIL":
            random.seed(seed)
            image = Image.fromarray(image)
            image = pil_prepare_augmentation(args)(image)
            image = np.array(image)
    if args.to_final_size:
        # Used especially for super resolution experiment
        size = (size[0], size[1])
    else:
        size = (image.shape[0], image.shape[1])
    if args.standardize_size:
        width = args.resize_gcd * round(size[0] / args.resize_gcd)
        height = args.resize_gcd * round(size[1] / args.resize_gcd)
        size = (width, height)
    # opencv will invert the width and height, need to confirm later
    # TODO: replace opencv with PIL
    image = cv2.resize(image, (size[1], size[0]))
    return image


def read_image(args, path, seed, size, ops=None, _to_tensor=True):
    """

    :param args:
    :param path:
    :param seed:
    :param size: size may be different especially for super-resolution research
    :param ops: callable functions, perform special options on images,
            ops can divide img into several patches in order to save memory
            ops can invert image color, switch channel
    :return:
    """
    if args.imgaug_engine == "cv2":
        if args.img_channel is 1:
            image = cv2.imread(path, 0)
        else:
            image = cv2.imread(path)
    elif args.imgaug_engine == "PIL":
        if args.img_channel is 1:
            image = Image.open(path).convert("L")
            image = np.array(image)
        else:
            image = Image.open(path)
            image = np.array(image)
    # image = misc.random_crop(image, args.crop_size, seed)
    if ops:
        image = ops(image, args, path, seed, size)
    # print("before: " + str(image.shape))
    if type(image) is list or type(image) is tuple:
        image = [prepare_image(args, img, seed, size) for img in image]
        image = [np.expand_dims(img, axis=-1) if len(img.shape) == 2 else img for img in image]
        if _to_tensor:
            return [to_tensor(args, img, seed, size, ops) for img in image]
        else:
            return image
    else:
        if len(image.shape) == 2:
            image = np.expand_dims(prepare_image(args, image, seed, size), axis=-1)
        else:
            image = prepare_image(args, image, seed, size)
        if _to_tensor:
            return to_tensor(args, image, seed, size, ops)
        else:
            return image


def to_tensor(args, image, seed, size, ops=None):
    trans = T.Compose([T.ToTensor(), T.Normalize(args.img_mean, args.img_std)])
    return trans(image)


def to_tensor_with_aug(args, image, seed, size, ops=None):
    if args.do_imgaug:
        imgaug.seed(seed)
        image = prepare_augmentation(args).augment_image(image)
    return to_tensor(args, image, seed, size, ops)


def just_return_it(args, data, seed, size, ops=None):
    """
    Because the label in cifar dataset is int
    So here it will be transfered to a torch tensor
    """
    return torch.tensor(data, dtype=torch.float)


def prepare_augmentation(args):
    aug_list = []
    if args.do_affine:
        aug_list.append(augmenters.Affine(scale={"x": args.scale, "y": args.scale},
                                          translate_percent={"x": args.translation, "y": args.translation},
                                          rotate=args.rotation, shear=args.shear, cval=args.aug_bg_color))
    if args.do_random_crop:
        aug_list.append(augmenters.Crop(px=args.crop_size, keep_size=args.keep_ratio, sample_independently=False))
    if args.do_random_flip:
        aug_list.append(augmenters.Fliplr(args.h_flip_prob))
        aug_list.append(augmenters.Flipud(args.v_flip_prob))
    if args.do_random_brightness:
        aug_list.append(augmenters.ContrastNormalization(args.brightness_vibrator))
        aug_list.append(augmenters.Multiply(args.brightness_multiplier, per_channel=0.2))
        #aug_list.append(augmenters.contrast.LinearContrast(alpha=args.linear_contrast))
    if args.do_random_noise:
        aug_list.append(augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))))
        aug_list.append(augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                         per_channel=0.5))
        
    seq = augmenters.Sequential(aug_list, random_order=False)
    return seq


def pil_prepare_augmentation(args):
    aug_list = []
    if args.do_affine:
        aug_list.append(T.RandomAffine(scale=args.scale, translate=args.translation,
                                       degrees=args.rotation, shear=args.shear, fillcolor=args.aug_bg_color))
    if args.do_random_crop:
        ratio = 1 if args.keep_ratio else (0.75, 1.33333333)
        # scale is augmented above so we will keep the scale here
        aug_list.append(T.RandomResizedCrop(size=args.crop_size, scale=1, ratio=ratio))
    if args.do_random_flip:
        aug_list.append(T.RandomHorizontalFlip(args.h_flip_prob))
        aug_list.append(T.RandomHorizontalFlip(args.v_flip_prob))
    if args.do_random_brightness:
        aug_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    return T.Compose(aug_list)


def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]
