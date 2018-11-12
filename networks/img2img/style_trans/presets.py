import os

def GeneralPattern_MAP(args):
    args.path = os.path.expanduser("~/Pictures/dataset/maps")
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
    return args

def UniquePattern_01(args):
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
    return args
    
PRESET = {
    "G_01": GeneralPattern_MAP,
    "U_01": UniquePattern_01
}
