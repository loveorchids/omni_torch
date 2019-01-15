def GeneralPattern_01(args):
    args.path = "~/Pictures/dataset/buddha"
    args.random_order_load = False
    args.loading_threads = 2
    args.img_channel = 1
    args.curr_epoch = 0
    
    args.do_imgaug = True
    # Affine Tranformation
    args.do_affine = False
    args.rotation = (-60, 60)
    # Random Crop
    args.do_random_crop = False
    # Random Zoom
    args.do_random_zoom = False
    args.pixel_eliminate = (100, 240)
    args.sample_independent = False
    # Random Flip
    args.do_random_flip = True
    args.v_flip_prob = 0.5
    args.h_flip_prob = 0.5
    # Random Color
    args.do_random_brightness = True
    args.brightness_vibrator = (0.8, 1.2),
    args.multiplier = (0.8, 1.2),
    args.per_channel_multiplier = 0.1,
    args.linear_contrast = 1.0,
    args.do_random_noise = False
    # Size
    args.to_final_size = False
    args.final_size = (320, 320)
    args.standardize_size = False
    args.standardize_gcd = 8
    
    return args


def UniquePattern_01(args):
    args.learning_rate = 1e-4
    args.weight_decay = 1e-4
    args.gpu_id = "1"
    args.cover_exist = True
    args.code_name = "cifar_torch"
    args.finetune = False
    
    args.S_MSE = False
    # Visualize the loss and gradient, update the loss distribution on every n epochs
    args.epoches_per_phase = 1
    args.steps_per_epoch = 100
    # The momentum when updating the los distribution
    args.loss_weight_momentum = 0.9
    args.loss_name = ["p_mse"]  # , "s_mse_1", "s_mse_2", "s_mse_3"]#, "s_mse_4", "s_mse_5"]
    args.measure_name = ["mae"]
    args.loss_weight = [1.0, 3.0, 5.0, 10.0]
    return args


def RuntimePattern(args):
    args.deterministic_train = True
    args.epoch_num = 300
    args.batch_size = 2
    
    args.img_mean = (0.0, 0.0, 0.0)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args


PRESET = {
    "general": GeneralPattern_01,
    "unique": UniquePattern_01,
    "runtime": RuntimePattern,
}