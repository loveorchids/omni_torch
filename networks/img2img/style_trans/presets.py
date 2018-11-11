import os, json

general_preset_name = "general_preset.json"
unique_preset_name = "unique_preset.json"


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def save_args(ops, args, name):
    path = os.path.expanduser(os.path.join(ops.path, ops.code_name, name))
    with open(path, "w") as file:
        json.dump(vars(args), file)
    return


def load_preset(args, path):
    path = os.path.expanduser(path)
    with open(path, "r") as file:
        data = json.load(file)
    data = vars(args).update(data)
    return Bunch(data)


class GeneralPattern_MAP:
    def create_args(self, ops):
        args = Bunch({})
        args.path = "~/Pictures/dataset/maps"
        args.deterministic_train = True
        args.epoch_num = 1000
        args.batch_size = 1
        
        args.img_channel = 3
        args.load_size = (600, 1200)
        args.do_resize = False
        args.do_imgaug = True
        args.do_affine = False
        args.do_random_crop = False
        args.crop_percent = ((0.0, 0.2), (0.0, 0.2), (0.0, 0.2), (0.0, 0.2))
        args.do_random_flip = True
        args.v_flip_prob = 0.5
        args.h_flip_prob = 0.5
        args.do_random_brightness = False
        args.do_random_noise = False
        
        args.img_mean = (0.5, 0.5, 0.5)
        args.img_std = (1.0, 1.0, 1.0)
        
        #args.segments = (6, 6)
        #args.segment_patch_size = (320, 320)
        save_args(ops, args, general_preset_name)
        return vars(args)
    
    def set(self, args):
        out = vars(args)
        tmp = self.create_args(args)
        out.update(tmp)
        return Bunch(out)


class UniquePattern_01:
    def create_args(self, ops):
        args = Bunch({})
        # The mean of curr_epoch is to increase the diversity during deterministic training
        # see code in set_arbitrary.py  => def __getitem__(self, index)
        args.curr_epoch = 0
        # Visualize the loss and gradient, update the loss distribution on every n epochs
        args.update_n_epoch = 5
        # The momentum when updating the los distribution
        args.loss_weight_momentum = 0.9
        args.loss_name = ["p_mse", "s_mse_1", "s_mse_2", "s_mse_3"]
        args.loss_weight = [0.65, 0.20, 0.10, 0.05]
        args.loss_weight_range = [(0.30, 1.00), (0.05, 0.50), (0.05, 0.50), (0.04, 0.40)]
        save_args(ops, args, unique_preset_name)
        return vars(args)
    
    def set(self, args):
        out = vars(args)
        out.update(self.create_args(args))
        return Bunch(out)


PRESET_Gen = {
    "01": GeneralPattern_MAP
}

PRESET_Unq = {
    "01": UniquePattern_01
}