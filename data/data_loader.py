<<<<<<< HEAD
import random
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import data.misc as misc
import data

IMGAUG_ENGINE = data.IMGAUG_ENGINE
ALLOW_WARNING = data.ALLOW_WARNING

if IMGAUG_ENGINE == "CV2":
    import imgaug
    from imgaug import augmenters
elif IMGAUG_ENGINE == "PIL":
    from PIL import Image

def prepare_image(args, image, seed, size):
    if args.do_imgaug:
        if IMGAUG_ENGINE == "CV2":
            imgaug.seed(seed)
            image = prepare_augmentation(args).augment_image(image)
            #image = misc.random_crop(image, args.crop_size, seed)
        elif IMGAUG_ENGINE == "PIL":
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
    image = cv2.resize(image, size)
    return image


def read_image(args, path, seed, size, ops=None):
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
    if args.img_channel is 1:
        image = cv2.imread(path, 0)
    else:
        image = cv2.imread(path)
    #image = misc.random_crop(image, args.crop_size, seed)
    if ops:
        image = ops(image)
    #print("before: " + str(image.shape))
    if type(image) is list or type(image) is tuple:
        image = [prepare_image(args, img, seed, size) for img in image]
        image = [np.expand_dims(img, axis=-1) if len(img.shape) == 2 else img for img in image]
        return [to_tensor(args, img, seed, size, ops) for img in image]
    else:
        if len(image.shape) == 2:
            image = np.expand_dims(prepare_image(args, image, seed, size), axis=-1)
        else:
            image = prepare_image(args, image, seed, size)
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
    if args.do_affine:
        aug_list.append(augmenters.Affine(scale={"x": args.scale, "y": args.scale},
                                      translate_percent={"x": args.translation, "y": args.translation},
                                      rotate=args.rotation, shear=args.shear, cval=args.aug_bg_color))
    if args.do_random_crop:
        aug_list.append(augmenters.Crop(px=args.crop_size, keep_size=args.keep_ratio))
    if args.do_random_flip:
        aug_list.append(augmenters.Fliplr(args.h_flip_prob))
        aug_list.append(augmenters.Flipud(args.v_flip_prob))
    if args.do_random_brightness:
        aug_list.append(augmenters.ContrastNormalization((0.75, 1.5)))
        aug_list.append(augmenters.Multiply((0.9, 1.1), per_channel=0.2))
    if args.do_random_noise:
        aug_list.append(augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))))
        aug_list.append(augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                         per_channel=0.5))
    seq = augmenters.Sequential(aug_list, random_order=True)
    return seq

def pil_prepare_augmentation(args):
    aug_list = []
    if args.do_affine:
        aug_list.append(T.RandomAffine(scale=args.scale, translate=args.translation,
                                      degrees=args.rotation, shear=args.shear, fillcolor =args.aug_bg_color))
    if args.do_random_crop:
        ratio = 1 if args.keep_ratio else (0.75, 1.33333333)
        # scale is augmented above so we will keep the scale here
        aug_list.append(T.RandomResizedCrop(size=args.loadsize, scale=1, ratio=ratio))
    if args.do_random_flip:
        aug_list.append(T.RandomHorizontalFlip(args.h_flip_prob))
        aug_list.append(T.RandomHorizontalFlip(args.v_flip_prob))
    if args.do_random_brightness:
        aug_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    return T.Compose(aug_list)

def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]
=======
import random
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import data.misc as misc
import data

IMGAUG_ENGINE = data.IMGAUG_ENGINE
ALLOW_WARNING = data.ALLOW_WARNING

if IMGAUG_ENGINE == "CV2":
    import imgaug
    from imgaug import augmenters
elif IMGAUG_ENGINE == "PIL":
    from PIL import Image

def prepare_image(args, image, seed, size):
    if args.do_imgaug:
        if IMGAUG_ENGINE == "CV2":
            imgaug.seed(seed)
            image = prepare_augmentation(args).augment_image(image)
            #image = misc.random_crop(image, args.load_size, seed)
        elif IMGAUG_ENGINE == "PIL":
            random.seed(seed)
            image = Image.fromarray(image)
            image = pil_prepare_augmentation(args)(image)
            image = np.array(image)
    if args.do_resize:
        size = (size[0], size[1])
    else:
        size = (image.shape[0], image.shape[1])
    ver = args.resize_gcd * round(size[0] / args.resize_gcd)
    height = args.resize_gcd * round(size[1] / args.resize_gcd)
    image = cv2.resize(image, (ver, height))
    return image


def read_image(args, path, seed, size, ops=None):
    if args.img_channel is 1:
        image = cv2.imread(path, 0)
    else:
        image = cv2.imread(path)
    #image = misc.random_crop(image, args.load_size, seed)
    if ops:
        image = ops(image)
    #print("before: " + str(image.shape))
    if type(image) is list or type(image) is tuple:
        image = [prepare_image(args, img, seed, size) for img in image]
        image = [np.expand_dims(img, axis=-1) if len(img.shape) == 2 else img for img in image]
        return [to_tensor(args, img, seed, size, ops) for img in image]
    else:
        if len(image.shape) == 2:
            image = np.expand_dims(prepare_image(args, image, seed, size), axis=-1)
        else:
            image = prepare_image(args, image, seed, size)
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
    if args.do_affine:
        aug_list.append(augmenters.Affine(scale={"x": args.scale, "y": args.scale},
                                      translate_percent={"x": args.translation, "y": args.translation},
                                      rotate=args.rotation, shear=args.shear, cval=args.aug_bg_color))
    if args.do_random_crop:
        aug_list.append(augmenters.Crop(px=args.load_size, keep_size=args.keep_ratio))
    if args.do_random_flip:
        aug_list.append(augmenters.Fliplr(args.h_flip_prob))
        aug_list.append(augmenters.Flipud(args.v_flip_prob))
    if args.do_random_brightness:
        aug_list.append(augmenters.ContrastNormalization((0.75, 1.5)))
        aug_list.append(augmenters.Multiply((0.9, 1.1), per_channel=0.2))
    if args.do_random_noise:
        aug_list.append(augmenters.Sometimes(0.2, augmenters.GaussianBlur(sigma=(0, 0.1))))
        aug_list.append(augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                         per_channel=0.5))
    seq = augmenters.Sequential(aug_list, random_order=True)
    return seq

def pil_prepare_augmentation(args):
    aug_list = []
    if args.do_affine:
        aug_list.append(T.RandomAffine(scale=args.scale, translate=args.translation,
                                      degrees=args.rotation, shear=args.shear, fillcolor =args.aug_bg_color))
    if args.do_random_crop:
        ratio = 1 if args.keep_ratio else (0.75, 1.33333333)
        # scale is augmented above so we will keep the scale here
        aug_list.append(T.RandomResizedCrop(size=args.loadsize, scale=1, ratio=ratio))
    if args.do_random_flip:
        aug_list.append(T.RandomHorizontalFlip(args.h_flip_prob))
        aug_list.append(T.RandomHorizontalFlip(args.v_flip_prob))
    if args.do_random_brightness:
        aug_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    return T.Compose(aug_list)

def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]
>>>>>>> d91fb3933ce89088cfee1fdb9a655c37cdea9bf4
