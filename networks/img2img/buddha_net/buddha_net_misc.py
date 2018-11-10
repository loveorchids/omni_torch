import os, math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

def update_loss_weight(losses, keys, pre, bound, momentum=0.8):
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
    assert 0 <= momentum  < 0.9999
    avg = [float(np.mean(loss)) for loss in losses]
    l = sum(avg)
    weight = [a/l for a in avg]
    avg = [pre[keys[i]] * momentum + weight[i] * (1 - momentum) for i, a in enumerate(avg)]
    # --------------- support up and low bound of weight -----------------
    mask = [out_of_bound(a, bound[keys[i]][0], bound[keys[i]][1]) for i, a in enumerate(avg)]
    if mask[0] == -1 and all(mask):
        pass
    else:
        s = sum([avg[i] if n is -1 else 0 for i, n in enumerate(mask)])
        remain = 1 - sum([0 if n is -1 else bound[keys[i]][n] for i, n in enumerate(mask)])
        weight = [avg[i] / s * remain if n is -1 else bound[keys[i]][n] for i, n in enumerate(mask)]
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
    
    plt.subplots(figsize=(18, 6))
    
    plt.plot("x", names[0], data=df, markersize=1, linewidth=1)
    plt.plot("x", names[1], data=df, markersize=2, linewidth=1)
    plt.plot("x", names[2], data=df, markersize=2, linewidth=1)
    plt.plot("x", names[3], data=df, markersize=3, linewidth=1)
    
    plt.legend(loc='upper right')
    img_name = name + str(epoch).zfill(4) + ".jpg"
    plt.savefig(os.path.join(save_path, img_name))
    # plt.show()
    plt.close()

def to_nd_image(tensor):
    assert len(tensor.shape) == 3
    img = tensor.data.to("cpu").numpy() * 255 + 128
    img = img.transpose((1, 2, 0)).astype("uint8")
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    return img

def normalize_grad_to_image(tensor):
    assert len(tensor.shape) == 3
    tensor = tensor.data
    min_v = torch.min(tensor)
    diff = torch.max(tensor) - min_v
    if diff == 0:
        tensor = torch.zeros(tensor.shape) + 255
    else:
        tensor = (tensor - min_v) / diff * 255
    array = tensor.to("cpu").numpy().astype("uint8")
    num = array.shape[0]
    if num in [1, 3]:
        # Plot the image directly
        # delete one dimension of this is a gray-scale image
        canvas = array.transpose((1, 2, 0))
    else:
        v = array.shape[1]
        h = array.shape[2]
        v_num = int(max(math.sqrt(num), 1))
        h_num = int(math.ceil(num / v_num))
        canvas = np.zeros((v_num * v, (h_num * h))).astype("uint8")
        for i in range(v_num):
            for j in range(h_num):
                if (i * h_num + j) >= num:
                    break
                canvas[i * v: (i + 1) * v, j * h: (j + 1) * h] = array[i * h_num + j, :, :]
    if len(canvas.shape) == 2:
        canvas = np.expand_dims(canvas, axis=-1)
    if canvas.shape[2] == 1:
        canvas = np.tile(canvas, (1, 1, 3))
    return canvas


def plot_cv2(tensor, path, op=to_nd_image, title=None, sub_title=None, ratio=1):
    #TODO
    """
    This is a function to plot one tensor at a time.
    :param tensor: can be gradient, parameter, data_batch, etc.
    :param op: operation that convert a (C, W, H) tensor into nd-array
    :param title:
    :param sub_title:
    :param ratio:
    :param path: if not None, function will save the image onto the local disk.
    :return:
    """
    assert 0.2 <= ratio  <= 5, "this ratio is too strange"
    num = tensor.size(0)
    v_num = int(max(math.sqrt(num) * ratio, 1))
    h_num = int(math.ceil(num / v_num))
    v_sub = math.ceil(math.sqrt(tensor.size(1)) * tensor.size(2) * v_num /100)
    h_sub = math.ceil(math.sqrt(tensor.size(1)) * tensor.size(3) * h_num / 100)
    fig, axis = plt.subplots(ncols=h_num, nrows=v_num, figsize = (h_sub, v_sub))
    if v_num == 1 and h_num == 1:
        cv2.imwrite(path, op(tensor[0]))
    else:
        for i in range(v_num):
            for j in range(h_num):
                axis[i * h_num + j].imshow(op(tensor[i * h_num + j]))
    if title:
        plt.title(title)
    if sub_title:
        plt.suptitle(sub_title)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def plot(tensor, op=to_nd_image, title=None, sub_title=None, ratio=1, path=None):
    """
    This is a function to plot one tensor at a time.
    :param tensor: can be gradient, parameter, data_batch, etc.
    :param op: operation that convert a (C, W, H) tensor into nd-array
    :param title:
    :param sub_title:
    :param ratio:
    :param path: if not None, function will save the image onto the local disk.
    :return:
    """
    assert 0.2 <= ratio  <= 5, "this ratio is too strange"
    num = tensor.size(0)
    if tensor.size(2) * tensor.size(3) <= 500:
        plt.axis("off")

    v_num = int(max(math.sqrt(num) * ratio, 1))
    h_num = int(math.ceil(num / v_num))
    v_sub = max(math.ceil(math.sqrt(tensor.size(1)) * tensor.size(2) * v_num /100), 4)
    h_sub = max(math.ceil(math.sqrt(tensor.size(1)) * tensor.size(3) * h_num / 100), 4)
    fig, axis = plt.subplots(ncols=h_num, nrows=v_num, figsize = (h_sub, v_sub))
    if v_num == 1 and h_num == 1:
        axis.imshow(op(tensor[0]))
    else:
        for i in range(v_num):
            for j in range(h_num):
                if len(axis.shape) == 1:
                    axis[i * h_num + j].imshow(op(tensor[i * h_num + j]))
                elif len(axis.shape) == 2:
                    print(i * h_num + j)
                    if (i * h_num + j) >= num:
                        break
                    axis[i, j].imshow(op(tensor[i * h_num + j]))
    if title:
        plt.title(title)
    if sub_title:
        plt.suptitle(sub_title)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
    
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

