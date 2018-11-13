import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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
    assert 0 <= momentum < 0.9999
    avg = [float(np.mean(loss)) for loss in losses]
    l = sum(avg)
    weight = [a / l for a in avg]
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
