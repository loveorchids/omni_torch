import random, warnings, itertools
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import omni_torch.data as data
import omni_torch.utils as util
import imgaug
from imgaug import augmenters

ALLOW_WARNING = data.ALLOW_WARNING

def read_image(args, path, seed, size, ops=None, _to_tensor=True):
    """
    Default image loading function invoked by Dataset object(Arbitrary, Img2Img, ILSVRC)
    :param args:
    :param path:
    :param seed:
    :param size: size may be different especially for super-resolution research
    :param ops: callable functions, perform special options on images,
            ops can divide img into several patches in order to save memory
            ops can invert image color, switch channel
    :return:
    """
    if args.img_channel is 1:
        image = cv2.imread(path, 0)
    else:
        image = cv2.imread(path)
    if args.normalize_img:
        image = cv2.normalize(image, None, args.normalize_min, args.normalize_max, cv2.NORM_MINMAX)
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

def prepare_image(args, image, seed, size):
    if args.do_imgaug:
        imgaug.seed(seed)
        image = prepare_augmentation(args).augment_image(image)
    if args.to_final_size:
        # Used especially for super resolution experiment
        size = (size[0], size[1])
    else:
        size = (image.shape[0], image.shape[1])
    if args.standardize_size:
        width = args.standardize_gcd * round(size[0] / args.standardize_gcd)
        height = args.standardize_gcd * round(size[1] / args.standardize_gcd)
        size = (width, height)
    # opencv will invert the width and height
    image = cv2.resize(image, (size[1], size[0]))
    return image

def to_tensor(args, image, seed, size, ops=None):
    image = util.normalize_image(args, image)
    trans = T.Compose([T.ToTensor()])
    return trans(image.astype("float32"))


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
    # -----------------------Create imgaug process from args------------------------
    aug_dict = {}
    if args.do_affine:
        aug_dict.update({"affine": [
            augmenters.Affine(scale={"x": args.scale_x, "y": args.scale_y},
                              translate_percent={"x": args.translation_x, "y": args.translation_y},
                              rotate=args.rotation, shear=args.shear, cval=args.aug_bg_color),
        ]})
    if args.do_random_crop:
        aug_dict.update({"random_crop": [
            augmenters.CropToFixedSize(width=args.crop_size[1], height=args.crop_size[0]),
        ]})
    if args.do_random_zoom:
        aug_dict.update({"random_zoom": [
            augmenters.Crop(px=tuple(args.pixel_eliminate), sample_independently=args.sample_independent),
        ]})
    if args.do_random_flip:
        aug_dict.update({"random_flip": [
            augmenters.Fliplr(args.h_flip_prob),
            augmenters.Flipud(args.v_flip_prob),
        ]})
    if args.do_random_brightness:
        aug_dict.update({"random_brightness": [
            augmenters.ContrastNormalization(args.brightness_vibrator),
            augmenters.Multiply(args.multiplier, per_channel=args.multiplier_per_channel),
            augmenters.LinearContrast(alpha=args.linear_contrast),
        ]})
    if args.do_random_noise:
        aug_dict.update({"random_noise": [
            augmenters.GaussianBlur(sigma=args.gaussian_sigma),
            #augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        ]})
    # ----------------------------------------------Combine imgaug process----------------------------------------------
    if type(args.imgaug_order) is list:
        assert len(set(args.imgaug_order)) == len(args.imgaug_order), \
            "repeated element in args.imgaug_order"
        assert len(args.imgaug_order) == len(aug_dict), \
            "args.imgaug_order has %s elements while aug_dict has %s elements"\
            %(len(args.imgaug_order, len(aug_dict)))
        try:
            aug_list = [aug_dict[item] for item in args.imgaug_order]
        except KeyError:
            print(aug_dict.keys())
            print("some element in args.imgaug_order does not match the key above")
            raise KeyError
        seq = augmenters.Sequential(list(itertools.chain.from_iterable(aug_list)))
    elif args.imgaug_order == "random":
        seq = augmenters.Sequential(list(itertools.chain.from_iterable(aug_dict.values())),
                                    random_order=True)
    elif args.imgaug_order == "default":
        seq = augmenters.Sequential(list(itertools.chain.from_iterable(aug_dict.values())),
                                    random_order=False)
    else:
        warnings.warn("unrecognizable args.amgaug_order, it should either be 'random', 'default' or a list.")
        seq = augmenters.Sequential(list(itertools.chain.from_iterable(aug_dict.values())),
                                    random_order=False)
    return seq


def pil_prepare_augmentation(args):
    aug_list = []
    if args.do_affine:
        aug_list.append(T.RandomAffine(scale=args.scale, translate=args.translation_x,
                                       degrees=args.rotation, shear=args.shear, fillcolor=args.aug_bg_color))
    if args.do_random_crop:
        ratio = 1 if args.keep_size else (0.75, 1.33333333)
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

if __name__ == "__main__":
    import os
    img_path = os.path.expanduser("~/Pictures/sample.jpg")

