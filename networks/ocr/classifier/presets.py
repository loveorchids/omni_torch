def GeneralPattern_OCR1(args):
    args.path = "~/Pictures/dataset/ocr"
    args.deterministic_train = False
    args.epoch_num = 50
    args.imgaug_engine = "cv2"
    args.batch_size = 1024
    args.random_order_load = False
    args.loading_threads = 6

    args.img_channel = 1

    args.do_imgaug = True
    # Affine Tranformation
    args.do_affine = False
    # Random Crop
    args.do_random_crop = False
    args.crop_size = (512, 512)
    args.keep_ratio = True
    # Random Flip
    args.do_random_flip = False
    args.v_flip_prob = 0.5
    args.h_flip_prob = 0.5
    # Random Color
    args.do_random_brightness = True
    args.brightness_vibrator = (0.7, 1.0)
    args.brightness_multiplier = (1.0, 2.0)
    args.linear_contrast = 1
    args.gamma_contrast=1.5
    args.do_random_noise = False
    # Size
    args.to_final_size = False
    args.final_size = (384, 384)
    args.standardize_size = False
    args.resize_gcd = 8

    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    return args

def Unique_Patterns_01(args):
    args.learning_rate = 2e-4
    args.curr_epoch = 0
    args.epoches_per_phase = 10
    args.resize_height = 32
    args.model_load_size = [40]
    args.output_size = 19
    args.seperator_offset = 2
    args.space_width = 16
    args.load_samples = 102400
    args.bad_distance = 2
    args.batch_norm = True
    args.load_img_offset = 2
    args.vibration = 2
    args.min_character_width = 5
    args.img_folder_name = ["segment", "normal"]
    args.img_folder_name_val = "test_set"
    args.txt_file_name_val = "comma.txt"
    args.bottleneck_size = 1536

    # Load Image
    args.stretch_img = True
    args.allow_left_aux = True
    args.left_aux_width = 2
    args.allow_right_aux = True
    args.right_aux_width = 2
    return args

def Runtime_Patterns(args):
    args.cover_exist = True
    args.code_name = "classify"
    args.gpu_id = "0"
    args.finetune = False
    return args

PRESET={
    "ocr1" : GeneralPattern_OCR1,
    "U_01": Unique_Patterns_01,
    "runtime": Runtime_Patterns,
}