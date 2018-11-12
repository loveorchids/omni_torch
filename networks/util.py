import os, glob
import torch
import json

def get_stride_padding_kernel(input, out):
    """
    Calculation based on:
    https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
    :param input:
    :param out:
    :return:
    """
    for K in range(1, 6):
        for S in range(1, 6):
            for P in range(6):
                if float(P) == (input - 1 - S * (out - 1)) / 2:
                    return S, P, K
                

    
def load_preset(args, preset, preset_code):
    class Bunch:
        def __init__(self, adict):
            self.__dict__.update(adict)
    if preset_code.endswith(".json"):
        assert os.path.exists(preset_code)
        path = os.path.expanduser(preset_code)
        with open(path, "r") as file:
            data = json.load(file)
        data = vars(args).update(data)
        return Bunch(data)
    else:
        args = preset[preset_code](args)
        #pattern_dict = pattern.create_args(self.args)
        return args

def save_args(args):
    path = os.path.expanduser(os.path.join(args.path, args.code_name, "preset.json"))
    with open(path, "w") as file:
        json.dump(vars(args), file)
    return

                
def save_model(args, epoch, state_dict, keep_latest=5):
    model_list = [_  for _ in glob.glob(args.model_dir + "/*.pth") if os.path.isfile(_)]
    model_list.sort()
    if len(model_list) < keep_latest:
        pass
    else:
        remove_list = model_list[: (keep_latest-1)*-1]
        for item in remove_list:
            os.remove(item)
    model_path = os.path.join(args.model_dir, "epoch_" + str(epoch).zfill(4)  + ".pth")
    torch.save(state_dict, model_path)
    
def load_latest_model(args, net):
    model_list = [_ for _ in glob.glob(args.model_dir + "/*.pth") if os.path.isfile(_)]
    model_list.sort()
    net.load_state_dict(torch.load(model_list[-1]))
    return net
    
if __name__ == "__main__":
    pass
    #save_model(1, 2, 3, os.path.expanduser("~/Documents/tmp"))
