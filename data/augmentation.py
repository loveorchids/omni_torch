import random, warnings, itertools
import imgaug
from imgaug import augmenters

def prepare_deterministic_augmentation(args, det_info):
    """
    Created each time when loading a specific image
    :param args:
    :param det_info:
    :return:
    """
    # --------------------Create deterministic process from args---------------------
    rotation, do_crop = None, None
    if det_info is None:
        return None
    if len(det_info) == 1:
        rotation = det_info
    elif len(det_info) == 4:
        do_crop = det_info
        top_crop, right_crop, bottom, left = det_info
    elif len(det_info) == 5:
        do_crop = det_info
        rotation, top_crop, right_crop, bottom, left = det_info
    else:
        return None
    aug_list = []
    if rotation:
        aug_list.append(
            augmenters.Affine(rotate=rotation, cval=args.aug_bg_color),
        )
    if do_crop:
        aug_list.append(
            augmenters.Crop(px=(top_crop, right_crop, bottom, left)),
        )
    return aug_list


def prepare_augmentation(args):
    """
    Created when declaring the data_loading pipeline
    :param args:
    :return:
    """
    aug_dict = {}
    default = ["affine", "crop", "pad", "flip", "brightness", "noise"]
    # --------------------------------------Geometry---------------------------------------
    if args.do_affine:
        aug_dict.update({"affine": [
            augmenters.Affine(scale={"x": args.scale_x, "y": args.scale_y},rotate=args.rotation,
                              translate_percent={"x": args.translation_x, "y": args.translation_y},
                              shear=args.shear, cval=args.aug_bg_color, name="rand_affine"),
        ]})
    if args.do_crop_to_fix_size:
        aug_dict.update({"crop": [
            augmenters.CropToFixedSize(width=args.crop_size[1], height=args.crop_size[0],
                                       name="crop_to_fix_size"),
        ]})
    if args.do_pad_to_fix_size:
        aug_dict.update({"pad": [
            augmenters.PadToFixedSize(width=args.padding_size[1], height=args.padding_size[0],
                                      pad_cval=args.aug_bg_color, position=args.padding_position,
                                      name="pad_to_fix_size"),
        ]})
    if args.do_random_flip:
        aug_dict.update({"flip": [
            augmenters.Fliplr(args.h_flip_prob, name="horizontal_flip"),
            augmenters.Flipud(args.v_flip_prob, name="vertical_flip"),
        ]})
    # -------------------------------Color and Brightness--------------------------------
    # TODO: consider how to add Sometimes, OneOf #01/02
    if args.do_random_brightness:
        aug_dict.update({"brightness": [
            augmenters.ContrastNormalization(args.brightness_vibrator),
            augmenters.Multiply(args.multiplier, per_channel=args.multiplier_per_channel),
            augmenters.LinearContrast(alpha=args.linear_contrast),
        ]})
    if args.do_random_noise:
        aug_dict.update({"noise": [
            augmenters.GaussianBlur(sigma=args.gaussian_sigma),
            #augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        ]})

    # ---------------------------------------Textures----------------------------------------
    # TODO: consider how to add Sometimes, OneOf #02/02
    
    # ---------------------------Combine Imgaug Process------------------------------
    if args.imgaug_order and type(args.imgaug_order) is list:
        # Remove repeated elements
        imgaug_order = list(set(args.imgaug_order))
        try:
            aug_list = [aug_dict[item] for item in imgaug_order]
            aug_list = list(itertools.chain.from_iterable(aug_list))
        except KeyError:
            not_contained = [key for key in imgaug_order if key not in aug_dict.keys()]
            print("%s in args.imgaug_order is not contained in the defined sequential: %s"
                  %(not_contained, aug_dict.keys()))
            raise KeyError
        if len(imgaug_order) != len(aug_dict):
            not_contained = [key for key in aug_dict.keys() if key not in imgaug_order]
            warnings.warn("You did not specify the whole sequential order for imgaug, \n"
                          "as the args.imgaug_order only has %s elements while aug_dict has %s elements, \n"
                          "underdetermined operations are: %s \n"
                          "omni_torch randomize the operations that does not contained in args.imgaug_order"
                          % (len(imgaug_order), len(aug_dict), not_contained))
            not_contained = [aug_dict[key] for key in  random.shuffle(not_contained)]
            aug_list = list(itertools.chain.from_iterable(not_contained))
            seq = list(itertools.chain.from_iterable(aug_dict.values())) + aug_list
        else:
            seq = aug_list
    else:
        if args.imgaug_order == "default":
            seq = []
            for item in default:
                try:
                    seq += aug_dict[item]
                except KeyError:
                    continue
        else:
            # perform random shuffle
            seq = list(itertools.chain.from_iterable(aug_dict.values()))
            seq = random.shuffle(seq)
    return seq


def combine_augs(det_list, rand_list, size):
    """
    :param det_list: represent for deterministic augmentation
    :param rand_list: represent for random augmentation
    :param size: represent for deterministic resize operation
    :return: imgaug.augmenters.Sequential Object
    """
    # ----------------------------------------------Combine imgaug process----------------------------------------------
    if det_list is None:
        det_list = []
    if rand_list is None:
        rand_list = []
    if size is None:
        size = []
    else:
        size = [augmenters.Resize(size=size)]
    if len(det_list) == len(rand_list) == len(size) == 0:
        return None
    return augmenters.Sequential(det_list + rand_list + size, random_order=False)
