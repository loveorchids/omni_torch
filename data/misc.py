import numpy as np
import warnings
import omni_torch.data as data

ALLOW_WARNING = data.ALLOW_WARNING

def number_to_char(num):
    assert (num >= 0 and num < 26), "Max 26 kind of input are supported."
    return chr(num+65)

def extension_check(file, extensions):
    if file[file.rfind(".")+1:] in extensions:
        return True
    else:
        return False

def get_shape(placeholder):
    try:
        result = [_ for _ in placeholder.shape.as_list() if _ is not None]
    except ValueError:
        result = ()
    return result

def compliment_dim(input, dim):
    assert len(input) <= dim, "length of input should not larger than dim."
    if len(input) == dim:
        return input
    else:
        repeat = dim // len(input)
        if type(input) is list:
            input = input * repeat + input[:dim - len(input) * repeat]
        elif type(input) is np.ndarray:
            input = np.concatenate((np.tile(input, repeat), input[:dim - len(input) * repeat]))
        return input
    
def str2bytes(str):
    return str.encode("ASCII")

def random_crop(image, crop_size, seed=None):
    crop_size = list(crop_size)
    RAISE_WARNING = False
    if crop_size[0] <= 0 or crop_size[1] <= 0:
        raise AssertionError("crop_size should bigger than 0")
    if image.shape[0] < crop_size[0]:
        crop_size[0] = image.shape[0]
        RAISE_WARNING = True
    if image.shape[1] < crop_size[1]:
        crop_size[1] = image.shape[1]
        RAISE_WARNING = True
    if RAISE_WARNING and ALLOW_WARNING:
        warnings.warn('One dimension of crop_size is larger than image itself.')
    if seed:
        np.random.seed(seed)
    w = np.random.randint(0, image.shape[0] - crop_size[0] + 1)
    h = np.random.randint(0, image.shape[1] - crop_size[1] + 1)
    if len(image.shape) == 2:
        # Grayscale Image
        return image[w:w + crop_size[0], h:h + crop_size[1]]
    elif len(image.shape) == 3:
        # RGB Image
        return image[w:w + crop_size[0], h:h + crop_size[1], :]
    else:
        raise TypeError("This function cannot process a tensor with a rank>=4")
        

if __name__ == "__main__":
    img = np.random.randint(0, 1, size=[20, 20])
    crop = random_crop(img, (20, 20))
    print(crop.shape)
    crop = random_crop(img, (15, 21))
    print(crop.shape)
    crop = random_crop(img, (20, 1))
    print(crop.shape)
    crop = random_crop(img, (21, 35))
    print(crop.shape)
    pass
    data.allow_warning(False)