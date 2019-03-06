import random, warnings, itertools
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import omni_torch.data as data
import omni_torch.utils as util
import omni_torch.data.augmentation as aug
import imgaug
from imgaug import augmenters

ALLOW_WARNING = data.ALLOW_WARNING

def read_image(args, items, seed, size, pre_process=None, rand_aug=None,
               bbox_loader=None, _to_tensor=True):
    """
    Default image loading function invoked by Dataset object(Arbitrary, Img2Img, ILSVRC)
    :param args:
    :param path:
    :param seed:
    :param size: size may be different especially for super-resolution research
    :param pre_process: callable functions, perform special options on images,
            ops can divide img into several patches in order to save memory
            ops can invert image color, switch channel, increase contrast
            ops can also calculate the infomation extactable brom image, e.g. affine matrix
    :return:
    """
    if bbox_loader:
        # image should be an np.ndarray
        # bbox should be an imgaug BoundingBoxesOnImage instance
        image, bbox, box_label = bbox_loader(args, items, seed, size)
    else:
        path = items
        if args.img_channel is 1:
            image = cv2.imread(path, 0)
        else:
            image = cv2.imread(path)
        bbox = None
    if pre_process:
        result = pre_process(image, args, path, seed, size)
        if type(result) is list or type(result) is tuple:
            image, data = result[0], result[1:]
        else:
            image, data = result, None
    else:
        data = None
    if args.standardize_size:
        if size is None:
            size = (image.shape[0], image.shape[1])
        # Sometimes we need the size of image to be dividable by certain number
        height = args.standardize_gcd * round(size[0] / args.standardize_gcd)
        width = args.standardize_gcd * round(size[1] / args.standardize_gcd)
        size = (height, width)
    # If pre-process returns some information about deterministic augmentation
    # Then initialize the deterministic augmentation based on that information
    det_aug_list = aug.prepare_deterministic_augmentation(args, data)
    aug_seq = aug.combine_augs(det_aug_list, rand_aug, size)
    if bbox:
        if aug_seq:
            # Do random augmentaion defined in pipline declaration
            aug_seq = aug_seq.to_deterministic()
            image = aug_seq.augment_image(image)
            bbox = aug_seq.augment_bounding_boxes([bbox])[0]
            #image_after = bbox.draw_on_image(image, thickness=2, color=[0, 0, 255])
            #cv2.imwrite("/home/wang/Pictures/tmp_after.jpg", image_after)
        # numpy-lize bbox
        coords = []
        labels = []
        h, w = image.shape[0], image.shape[1]
        for i, bbox in enumerate(bbox.bounding_boxes):
            condition_1 = bbox.x1 <= 0 and bbox.x2 <= 0
            condition_2 = bbox.y1 <= 0 and bbox.y2 <= 0
            condition_3 = bbox.x1 >= w -1 and bbox.x2 >= w -1
            condition_4 = bbox.y1 >= h -1 and bbox.y2 >= h -1
            if condition_1 or condition_2 or condition_3 or condition_4:
                # After aigmentation, at least one dimension of the bbox exceeded the image
                # omni_torch will ignore this bbox
                continue
            horizontal_constrain = lambda x: max(min(w, x), 0)
            vertival_constrain = lambda y: max(min(h, y), 0)
            coords.append([horizontal_constrain(bbox.x1)/h, vertival_constrain(bbox.y1)/w,
                           horizontal_constrain(bbox.x2)/h, vertival_constrain(bbox.y2)/w])
            labels.append(box_label[i])
        coords = torch.Tensor(coords)
        labels = torch.Tensor(labels)
    else:
        # With no bounding boxes, augment the image only
        if aug_seq:
            aug_seq = aug_seq.to_deterministic()
            image = aug_seq.augment_image(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if _to_tensor:
        if bbox:
            return to_tensor(args, image, seed, size), coords, labels
        return to_tensor(args, image, seed, size)
    else:
        if bbox:
            return image, coords, labels
        return image


def to_tensor(args, image, seed, size):
    image = util.normalize_image(args, image)
    trans = T.Compose([T.ToTensor()])
    return trans(image.astype("float32"))


def to_tensor_with_aug(args, image, seed, size, rand_aug):
    if args.do_imgaug:
        imgaug.seed(seed)
        image = rand_aug.augment_image(image)
    return to_tensor(args, image, seed, size)


def just_return_it(args, data, seed, size):
    """
    Because the label in cifar dataset is int
    So here it will be transfered to a torch tensor
    """
    return torch.tensor(data, dtype=torch.float)


def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]


if __name__ == "__main__":
    import os
    img_path = os.path.expanduser("~/Pictures/sample.jpg")

