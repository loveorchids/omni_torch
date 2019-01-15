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
import networks.ocr.attention_mnist.presets as preset
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

    #width, height = Image.open(path).size
    height, width, channel = cv2.imread(path).shape
    ratio = height / args.resize_height

    assert len(label) == len(coords)
    coords = [[0, 0, 0, 0]] + coords
    coords = (np.array(coords) / ratio).astype("int")
    return path, coords, num, label

def resize_height(image, args, path, seed, size):
    if image is None:
        print(path)
    width, height = image.shape[1], image.shape[0]
    width = round(width / height * args.resize_height)
    image = cv2.resize(image, (width, args.resize_height))
    return image

def get_path_and_label(args, length, paths, foldername):
    label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
                  '-': 12, '8': 13, '5': 14, '': 15, "/":16, "(":17, ")":18}
    with open(paths, "r", encoding="utf-8") as txtfile:
        output_path = []
        output_label = []
        for _, line in enumerate(txtfile):
            if _ in [7]:
                print("")
            output_path.append(os.path.join(args.path, foldername, line[: line.find(args.text_seperator)]))
            label = line[line.find(args.text_seperator) + 1:-1]
            new_label = []
            for char in label:
                if char == " ":
                    continue
                elif char == "\\":
                    char = "¥"
                new_label.append(char)
            try:
                output_label.append([label_dict[_] for _ in new_label])
            except KeyError:
                print("Key Error Occured at line %s"%(_))
    return [list(zip(output_path, output_label))]

def read_img_and_label(args, path, seed, size, ops=None):
    path, label = path[0], path[1]
    img_tensor = loader.read_image(args, path, seed, size, ops)
    if len(label) == 0:
        label = [99]
    return img_tensor, loader.just_return_it(args, label, seed, size)

def load_bbox_from_txt(args, length, paths, foldername):
    label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
                  '-': 12, '8': 13, '5': 14, '': 15, "/":16, "(":17, ")":18}
    width = args.model_load_size[-1]
    ignored_chars, white_spaces, total = 0, 0, 0
    output_path, output_label, output_seg, output_numbers = [], [], [], []
    with open(paths, "r", encoding="utf-8") as txtfile:
        for _, line in enumerate(txtfile):
            if _ >= args.load_samples:
                sum+=1
                break
            # Line looks like:
            # name, label, [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            path, coords, num, label = parse_line(args, line, foldername)
            #coords.pop(0)
            gt, end_line = [], []
            total += len(label)
            for i in range(len(label)):
                d = coords[i + 1][0] - coords[i][2]
                w = coords[i + 1][2] - coords[i + 1][0]
                x_start = coords[i][2]
                if w + int(args.space_width/2) > width and w+d > width:
                    ignored_chars +=1
                    print("At %s th position of %s has bad char of %s px width and  the distance before it is %s px."%(i, path, w, d))
                    # Continue? or Break?
                    continue
                _n = round(d / args.space_width)
                for n in range(_n):
                    white_spaces += 1
                    space_width = int(d / _n)
                    x1=random.randint(x_start + n * space_width, x_start + n * space_width+2)
                    x2 = random.randint(x_start + (n+1) * space_width - 2, x_start + (n+1) * space_width)
                    y1, y2 = random.randint(0, 4), random.randint(args.resize_height - 4, args.resize_height)
                    end_line.append([x1, x2, y1, y2])
                    gt.append(label_dict[''])
                gt.append(label_dict[label[i]])
                #end_line.append(coords[i + 1][2])
            output_path.append(path)
            output_numbers.append(num)
            output_label.append(gt)
            output_seg.append(end_line)
        return

def load_file_from_txt(args, length, paths, foldername):
    label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
                  '-': 12, '8': 13, '5': 14, '': 15, "/":16, "(":17, ")":18}
    with open(paths, "r", encoding="utf-8") as txtfile:
        output_path, output_label, output_seg, output_numbers = [], [], [], []
        sum = 0
        dif = args.model_load_size[-1] - args.resize_height
        h = args.resize_height
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
                if w + d > args.resize_height and d < dif:
                    pass
                    #print(w + d)
                    #print("Warning: extreme wide character encountered!!")
                _n = 0
                space_width = args.space_width
                while (w + d > args.resize_height and d > dif):
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
        return [output_path, output_label, output_seg, output_numbers]

def get_all_info(args, length, paths, dig_level=0):
    path, label, seg, numbers = load_file_from_txt(args, length, paths, dig_level)
    output_path_seg = list(zip(path, seg))
    return [output_path_seg]#, label, numbers]

def read_seg_info(args, path, seed, size, ops=None):
    width = args.model_load_size[-1]
    path, coord = path[0], path[1]
    #if path == "/home/wang/Pictures/dataset/ocr/many_font/15901.jpg":
        #print("target")
    img_tensor = loader.read_image(args, path, seed, size, ops)
    idx = random.randint(0, len(coord) - 2)
    width_end = min(int(coord[idx]) + width + args.load_img_offset, img_tensor.size(2))
    img = img_tensor[:, :, int(coord[idx]) + args.load_img_offset: width_end]
    try:
        if img.size(2) < width:
            # Make the background as white
            bg = (torch.zeros((args.img_channel, args.resize_height, width)) + 0.49)
            bg[:, :, :img.size(2)] = img
            img = bg
    except RuntimeError:
        print(path)
    #one_hot = loader.one_hot(args.model_load_size[-1], int(coord[idx+1]) - int(coord[idx]) -1)
    #return img, one_hot
    #print(path)
    return img, torch.tensor(int(coord[idx+1]) - int(coord[idx]))
    
def fetch_segment_data(args, sources, batch_size, foldername, shuffle=True):
    data = Arbitrary_Dataset(args, sources=sources, modes=[get_all_info],
                             load_funcs=[read_seg_info] + [loader.just_return_it]*2, dig_level=[foldername],
                             loader_ops=[resize_height, None, None])
    data.prepare()

    train_idx = [_ for _ in range(len(data.dataset) - 1) if _ % 3 != 0]
    validation_idx = [_ for _ in range(len(data.dataset) - 1) if _ % 3 == 0]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                               sampler=train_sampler, **kwargs)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                                    sampler=validation_sampler, **kwargs)
    return train_loader, validation_loader

def fetch_data(args, sources, batch_size, shuffle=False, pin_memory=False, txt_file=None):
    if sources[0] in ["open_cor"]:
        args.text_seperator = "\t"
    elif sources[0] in ["test_set"]:
        args.text_seperator = ","
    if txt_file:
        read_source = txt_file
    else:
        read_source = [_ + ".txt" for _ in sources]
    data = Arbitrary_Dataset(args, sources=read_source, modes=[get_path_and_label],
                             load_funcs=[read_img_and_label], dig_level=sources,
                             loader_ops=[resize_height])
    data.prepare()
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': pin_memory} if torch.cuda.is_available() else {}
    dataset = torch.utils.data.DataLoader(data, batch_size=batch_size,  shuffle=shuffle, **kwargs)
    return dataset

def fetch_hybrid_data(args, sources, foldername, batch_size=1, shuffle=True, pin_memory=False):
    data = []
    for i, source in enumerate(sources):
        datum = Arbitrary_Dataset(args, sources=[source + ".txt"], modes=[load_bbox_from_txt],
                                      load_funcs=[read_seg_info] , dig_level=[foldername[i]],
                                      loader_ops=[resize_height])
        datum.prepare()
        data.append(datum)

    samples = sum([len(d) for d in data]) - 1
    train_idx = [_ for _ in range(samples) if _ % 3 != 0]
    validation_idx = [_ for _ in range(samples) if _ % 3 == 0]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    workers = mpi.cpu_count() if args.loading_threads > mpi.cpu_count() \
        else args.loading_threads
    kwargs = {'num_workers': workers, 'pin_memory': pin_memory} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data),
                                               batch_size=batch_size, shuffle=shuffle,
                                               sampler=train_sampler, **kwargs)
    validation_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data),
                                                    batch_size=batch_size, shuffle=shuffle,
                                                    sampler=validation_sampler, **kwargs)
    return train_loader, validation_loader

def draw_multi_line(tensor, coord=None, gt=None):
    height = tensor.size(2)
    if tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    coord_color = (np.array([48, 33, 255]) - 128) / 255  # Red
    gt_color = (np.array([48, 255, 12]) - 128) / 255  # Green
    coord_color = np.expand_dims(np.expand_dims(np.expand_dims(coord_color, 0), -1), -1)
    coord_color = torch.tensor(np.tile(coord_color, (1, 1, height, 1)))
    gt_color = np.expand_dims(np.expand_dims(np.expand_dims(gt_color, 0), -1), -1)
    gt_color = torch.tensor(np.tile(gt_color, (1, 1, height, 1)))
    if coord:
        for i in coord:
            sep = min(max(i, 1), tensor.size(3) - 1)
            try:
                tensor[:, :, :, sep:sep + 1] = coord_color
            except RuntimeError:
                pass
    if gt:
        for j in gt:
            sep = min(max(j, 0), tensor.size(3) - 1)
            try:
                tensor[:, :, :, sep:sep+1] = gt_color
            except RuntimeError:
                pass
    return tensor

def draw_line(tensor, coord=None, gt=None):
    height = tensor.size(2)
    if tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    coord_color = (np.array([48, 33, 255]) - 128) / 255  # Red
    gt_color =( np.array([48, 255, 12]) - 128) / 255   # Green
    coord_color = np.expand_dims(np.expand_dims(np.expand_dims(coord_color, 0), -1), -1)
    coord_color = torch.tensor(np.tile(coord_color, (1, 1, height, 1)))
    gt_color = np.expand_dims(np.expand_dims(np.expand_dims(gt_color, 0), -1), -1)
    gt_color = torch.tensor(np.tile(gt_color, (1, 1, height, 1)))
    # Create a line of width 2
    if coord:
        coord = min(max(coord, 0), tensor.size(3)-1)
        tensor[:, :, :, coord:coord + 1] = coord_color
    if gt:
        tensor[:, :, :, gt - 1:gt] = gt_color
    return tensor

def to_gaussian_prob(length, mean, var=0.5):
    mean = mean.unsqueeze(-1).repeat(1, length).float()
    tensor = torch.arange(length).float().unsqueeze(0).repeat(mean.size(0), 1)
    return torch.tensor(2.71828).pow(-0.5 * ((tensor - mean) / var) ** 2)  * (0.39894 / var)

if __name__ == "__main__":
    a = to_gaussian_prob(32, [15, 5], [1, 1])
    args = BaseOptions().initialize()
    args.general_options = "ocr1"
    args.unique_options = "U_01"
    args.runtime_options = "runtime"

    args = util.prepare_args(args, preset.PRESET)
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    test_set = fetch_data(args, sources=["test_set"], batch_size=1)

    

            
        
