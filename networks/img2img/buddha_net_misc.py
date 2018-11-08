import os, math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

def update_loss_weight(losses, keys, pre, range, momentum=0.8):
    def out_of_bound(num, low_bound, up_bound):
        if num < low_bound:
            # return low bound index
            return 0
        elif num > up_bound:
            # return up bound index
            return 1
        else:
            return -1
    assert len(losses) == len(keys)
    assert momentum >= 0 and momentum < 0.9999
    avg = [float(np.mean(loss)) for loss in losses]
    l = sum(avg)
    weight = [a/l for a in avg]
    avg = [pre[keys[i]] * momentum + weight[i] * (1 - momentum) for i, a in enumerate(avg)]
    # --------------- support up and low bound of weight -----------------
    mask = [out_of_bound(a, range[keys[i]][0], range[keys[i]][1]) for i, a in enumerate(avg)]
    if mask[0] == -1 and all(mask):
        pass
    else:
        s = sum([avg[i] if n is -1 else 0 for i, n in enumerate(mask)])
        remain = 1 - sum([0 if n is -1 else range[keys[i]][n] for i, n in enumerate(mask)])
        weight = [avg[i] / s * remain if n is -1 else range[keys[i]][n] for i, n in enumerate(mask)]
    # --------------- support up and low bound of weight -----------------
    current = dict(zip(keys, weight))
    print(current)
    return current

def plot_loss_distribution(losses, keyname, save_path, name, epoch, weight):
    names = []
    for key in keyname:
        names.append(key.ljust(8) + ": " + str(weight[key])[:5])
    x_axis = range(len(losses[0]))
    losses.append(np.asarray(list(x_axis)))
    names.append("x")
    plot_data = dict(zip(names, losses))
    
    # sio.savemat(os.path.expanduser(args.path)+"loss_info.mat", plot_data)
    df = pd.DataFrame(plot_data)
    
    fig, axis = plt.subplots(figsize=(18, 6))
    
    plt.plot("x", names[0], data=df, markersize=1, linewidth=1)
    plt.plot("x", names[1], data=df, markersize=2, linewidth=1)
    plt.plot("x", names[2], data=df, markersize=2, linewidth=1)
    plt.plot("x", names[3], data=df, markersize=3, linewidth=1)
    
    plt.legend(loc='upper right')
    img_name = name + str(epoch).zfill(4) + ".jpg"
    plt.savefig(os.path.join(save_path, img_name))
    # plt.show()

def to_nd_image(tensor):
    img = tensor.data.to("cpu").numpy() * 255 + 128
    img = img.transpose((1, 2, 0)).astype("uint8")
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    return img

def normalize_grad_to_image(tensor):
    tensor = tensor.data
    min_v = torch.min(tensor)
    diff = torch.max(tensor) - min_v
    if diff == 0:
        tensor_norm = torch.zeros(tensor.shape) + 255
    else:
        tensor_norm = (tensor - min_v) / diff * 255
    tensor = tensor_norm.to("cpu").numpy().squeeze

def plot(tensor, dim=0, op=to_nd_image, visualize="plot"):
    """
    This is a function to plot one tensor at a time.
    :param tensor: can be gradient, parameter, data_batch, etc.
    :param axis: decomposite a tensor along n-th dimension
    :param op: operation that convert a (C, W, H) tensor into nd-array
    :return:
    """
    num = tensor.size(dim)
    v_num = int(math.sqrt(num))
    h_num = math.ceil(num / v_num)
    fig, axis = plt.subplots(v_num, h_num, figsize=(8, 8))
    if v_num == 1 and h_num == 1:
        axis.imshow(op(tensor[0]))
    else:
        for i in range(v_num):
            for j in range(h_num):
                axis[i, j].imshow(op(tensor[i * h_num + j]))
    plt.show()
    
def save_tensor_to_img(args, tensor_list, epoch):
    vertical_slice = []
    horizontal_slice = []
    for idx, tensor in enumerate(tensor_list):
        horizontal_slice.append(tensor.numpy().squeeze() * 255)
        if idx is not 0 and (idx+1) % args.segments[1] == 0:
            vertical_slice.append(np.concatenate(horizontal_slice, axis=1))
            horizontal_slice = []
    imgs = np.concatenate(vertical_slice, axis=0)
    cv2.imwrite(os.path.join(os.path.expanduser(args.log_dir), str(epoch)+".jpg"), cv2.bitwise_not(imgs))


def vis_image(args, tensorlist, epoch, batch, loss_dict, idx=None, concat_axis=1):
    if type(idx) is int:
        idx = [idx]
    if not idx:
        idx = range(max([t.size(0) for t in tensorlist]))
    for i in idx:
        img = np.concatenate([t[i].data.to("cpu").numpy().squeeze() * 255 + 128
                              for t in tensorlist], axis=concat_axis)
        img = img.astype("uint8")
        img_name = str(epoch)+"_"+str(batch)+"_"+str(i)+".jpg"
        path = os.path.join(os.path.expanduser(args.log_dir), img_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = math.sqrt(img.size) / 1000
        font_position = (img.shape[1] - int(700 * font_size * 2), img.shape[0] - int(20 * font_size * 2))
        thickness = round(font_size * 2)
        loss = ""
        keys = list(loss_dict.keys())
        keys.sort()
        for key in keys:
            loss += key.ljust(8) + ": " + str(loss_dict[key])[:6] + "  "
        if len(img.shape) == 2:
            img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))
        cv2.putText(img, loss, font_position, font, font_size, (48, 33, 255), thickness)
        cv2.imwrite(path, img)

