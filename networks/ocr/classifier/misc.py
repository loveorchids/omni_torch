import os, time, sys, math, random
sys.path.append(os.path.expanduser("~/Documents/omni_torch/"))
import torch, cv2
from torch.utils.data.sampler import SubsetRandomSampler
import multiprocessing as mpi
import numpy as np
from data.set_arbitrary import Arbitrary_Dataset
import networks.util as util
import data.path_loader as mode
import data.data_loader as loader
from options.base_options import BaseOptions
import networks.ocr.classifier.presets as preset
from PIL import Image

def parse_line(args, line, foldername):
    name_label = line[:line.rfind(":")]
    num = int(name_label[:name_label.find(".")])

    if num in []:
        print("PROBLEM")
    path = os.path.join(args.path, foldername, name_label[: name_label.find(":")])
    label = name_label[name_label.find(":") + 1:].strip()
    coords = line[line.rfind(":") + 1:]
    coords = coords.replace('[', '').split('],')
    coords = [list(map(int, s.replace(']', '').split(','))) for s in coords]

    width, height = Image.open(path).size
    ratio = height / args.resize_height

    assert len(label) == len(coords)
    coords = [[0, 0, 0, 0]] + coords
    coords = (np.array(coords) / ratio).astype("int")
    return path, coords, num, label

def resize_height(image, args, path, seed, size):
    width, height = image.shape[1], image.shape[0]
    width = round(width / height * args.resize_height)
    image = cv2.resize(image, (width, args.resize_height))
    return image

def resize_height_auto_contrast(image, args, path, seed, size):
    width, height = image.shape[1], image.shape[0]
    width = round(width / height * args.resize_height)
    image = cv2.resize(image, (width, args.resize_height))
    #image[np.where(image<)]
    #assert len(image.shape) == 2
    #image = image * 0.7 + cv2.equalizeHist(image)*0.3
    return image

def load_file_from_txt(args, length, paths, foldername):
    label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
                  '-': 12, '8': 13, '5': 14, '': 15, "/":16, "(":17, ")":18}
    with open(paths, "r", encoding="utf-8") as txtfile:
        output_path, output_label, output_seg, output_numbers = [], [], [], []
        sum = 0
        dif = args.model_load_size[-1] - args.resize_height
        for _, line in enumerate(txtfile):
            if _ >= args.load_samples:
                break
            # Line looks like:
            # name, label, [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            path, coords, num, label = parse_line(args, line, foldername)
            gt = []
            end_line = [0]
            sum += len(label)
            ignore_line = False
            for i in range(len(label)):
                d = coords[i + 1][0] - coords[i][2]
                w = coords[i + 1][2] - coords[i + 1][0]
                if w > args.resize_height:
                    ignore_line = True
                    break
                _n = 0
                #############################################################
                # Here we allow the random space size in order to make the space width looks
                # more likely to the real samples
                #############################################################
                while (w + d > args.resize_height and d > dif):
                    space_width = random.randint(4, args.space_width)
                    if d < space_width:
                        # add a seperator
                        end_line.append(coords[i][2] + d + _n * space_width)
                        d = 0
                    else:
                        # add a seperator
                        end_line.append(coords[i][2] + (_n + 1) * space_width)
                        d -= space_width
                        _n += 1
                    gt.append(label_dict[''])
                gt.append(label_dict[label[i]])
                end_line.append(coords[i + 1][2])
            if ignore_line:
                continue
            output_numbers.append(num)
            output_path.append(path)
            output_label.append(gt)
            output_seg.append(end_line)
        print(sum)
        output_path_seg = list(zip(output_path, output_seg, output_label))
        return [output_path_seg]

def load_coord_from_txt(args, length, paths, foldername):
    label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
                  '-': 12, '8': 13, '5': 14, '': 15, "/": 16, "(": 17, ")": 18}
    with open(paths, "r", encoding="utf-8") as txtfile:
        output_path, output_label, output_seg, output_numbers = [], [], [], []
        for _, line in enumerate(txtfile):
            if _ >= args.load_samples:
                break
            path, coords, num, label = parse_line(args, line, foldername)
            words, gt = [], []
            for i, coord in enumerate(coords):
                if i == 0:
                    continue
                words.append([coord[0], coord[2]])
                gt.append(label_dict[label[i-1]])
            output_path.append(path)
            output_label.append(gt)
            output_seg.append(words)
        output_path_seg = list(zip(output_path, output_seg, output_label))
        return [output_path_seg]

def get_slice(args, coord, label, image):
    width = args.model_load_size[-1]
    coord = [[coord[i], coord[i + 1]] for i in range(len(coord) - 1)]
    i = random.randint(0, len(coord) - 1)
    end = image.shape[1] if i + 1 == len(coord) else coord[i + 1][0]
    start = 0 if i == 0 else coord[i - 1][1]
    if label[i] in [2, 0, 9, 15, 16, 17, 18]:
        left_vibrator, right_vibrator = random.randint(0, 2), 0
    else:
        left_vibrator = random.randint(-1 * args.vibration, args.vibration)
        right_vibrator = random.randint(0, args.vibration)
    right_bound = min(image.shape[1], coord[i][0] + width + right_vibrator, end + right_vibrator)
    left_bound = max(0, coord[i][1] - width + left_vibrator, start + left_vibrator)
    width_end = random.randint(min(coord[i][1], right_bound), max(coord[i][1], right_bound))
    width_start = random.randint(min(coord[i][0], left_bound), max(coord[i][0], left_bound))
    if width_end - width_start < args.min_character_width:
        # this is the least distance between start line and end line
        if width_end < image.shape[1] - args.min_character_width:
            width_end += (args.min_character_width - width_end + width_start)
        else:
            width_start -= (args.min_character_width - width_end + width_start)
    return width_start, width_end, i

def read_single_character(args, path, seed, size, ops=None):
    path, coord, label = path[0], path[1], path[2]
    image = loader.read_image(args, path, seed, size, ops, _to_tensor=False)
    width_start, width_end, i = get_slice(args, coord, label, image)
    img = image[:, width_start: width_end, :]
    if args.allow_left_aux:
        if width_start - args.left_aux_width >= 0:
            left_aux = image[:, width_start - args.left_aux_width:width_start, :]
            img = np.concatenate((left_aux.astype("uint8"), img.astype("uint8")), axis=1)
    else:
        args.left_aux_width = 0
    if args.allow_right_aux:
        if width_end + args.right_aux_width <= image.shape[1]:
            right_aux = image[:, width_end : width_end+args.right_aux_width, :]
            img = np.concatenate((img.astype("uint8"), right_aux.astype("uint8")), axis=1)
    else:
        args.right_aux_width = 0
    try:
        if args.stretch_img:
            # Fit the image size to network by resize
            img = cv2.resize(img, (args.resize_height, args.resize_height))
        else:
            # Fit the image size to network by add a white patch on the left(if it can be added)
            if img.shape[1] > args.resize_height:
                # If the img width is larger than the network input size
                img = cv2.resize(img, (args.resize_height, args.resize_height))
            else:
                complement = np.zeros((args.resize_height, args.resize_height - img.shape[1], img.shape[2])) + 255
                img = np.concatenate((complement, img), axis=1).astype("uint8")
    except cv2.error:
        print("Accident happens in cv2::resize, on %s"%(path))
        img = np.zeros((args.resize_height, args.resize_height)).astype("uint8")
    # Add noises and lines
    img = add_noise_and_black_line(img, vertical=label[i] == 15)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    return loader.to_tensor(args, img, seed, size, ops), torch.tensor(label[i])

def add_noise_and_black_line(img, vertical=False):
    rand_num = random.uniform(-1, 1)
    color = random.randint(10, 120)
    if rand_num > 0:
        offset = int(rand_num * 4)
        if vertical:
            if rand_num < 0.8:
                # Add some noise image onto the white space
                pass
            # Then draw a vertical line
            k = [3, 4, 5, 6, 7]
            kernel_size = k[random.randint(0, 4)]
            img = cv2.line(img, (img.shape[1]-1-offset, 0), (img.shape[1]-1-offset, img.shape[0]), (color, color, color), int(rand_num*6)+2)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size / kernel_size
            img = cv2.filter2D(img, -1, kernel)
        else:
            if rand_num < 0.7:
                # draw line on the bottom
                img = cv2.line(img, (0, img.shape[0]-1-offset), (img.shape[1], img.shape[0] -1-offset), (color, color, color), max(int(rand_num*4), 1))
            if rand_num > 0.3:
                # Draw line on the top
                img = cv2.line(img, (0, offset), (img.shape[1], offset), (color, color, color), max(int(rand_num*4)-1, 1))
    return img

def fetch_classification_data(args, sources, foldername, batch_size=1, shuffle=False, pin_memory=False, split_val=True):
    data = []
    for i, source in enumerate(sources):
        datum = Arbitrary_Dataset(args, sources=[source + ".txt"], modes=[load_file_from_txt],
                                  load_funcs=[read_single_character], dig_level=[foldername[i]],
                                  loader_ops=[resize_height_auto_contrast])
        datum.prepare()
        data.append(datum)

    samples = sum([len(d) for d in data]) - 1
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': pin_memory} if torch.cuda.is_available() else {}
    if split_val:
        train_sampler = SubsetRandomSampler([_ for _ in range(samples) if _ % 4 != 0])
        validation_sampler = SubsetRandomSampler([_ for _ in range(samples) if _ % 4 == 0])
        train_set = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size,
                                              shuffle=shuffle,  sampler=train_sampler, **kwargs)
        val_set = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size,
                                              shuffle=shuffle,  sampler=validation_sampler, **kwargs)
        return train_set, val_set
    else:
        train_set = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size, shuffle=shuffle)
        return train_set

if __name__ == "__main__":
    args = BaseOptions().initialize()
    args.general_options = "ocr1"
    args.unique_options = "U_01"
    args.runtime_options = "runtime"

    args = util.prepare_args(args, preset.PRESET)
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    train_set, val_set = fetch_classification_data(args, sources=args.img_folder_name,
                                           foldername=args.img_folder_name, batch_size=4)
    for batch_idx, data in enumerate(train_set):
        print(data.shape)
    

            
        
