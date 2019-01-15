def GeneralPattern_OCR1(args):
    args.path = "~/Pictures/dataset/ocr"
    args.deterministic_train = False
    args.epoch_num = 30
    args.imgaug_engine = "cv2"
    args.batch_size = 200
    args.random_order_load = False
    args.loading_threads = 6

    args.img_channel = 1

    args.do_imgaug = False
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
    args.brightness_vibrator = 0.1
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
    args.learning_rate = 1e-5
    args.curr_epoch = 0
    args.epoches_per_phase = 5
    args.space_width = 16
    args.resize_height = 32
    args.model_load_size = [40]
    args.seperator_offset = 2
    args.load_samples = 50000
    args.bad_distance = 2
    args.batch_norm = True
    args.load_img_offset = 2
    args.img_folder_name = ["many_font"]
    args.img_folder_name_val = ["open_cor"]
    args.text_seperator = ","
    args.bottleneck_size = 3840
    return args

def Runtime_Patterns(args):
    args.cover_exist = True
    args.code_name = "overall"
    args.gpu_id = "0"
    args.finetune = True
    return args

PRESET={
    "ocr1" : GeneralPattern_OCR1,
    "U_01": Unique_Patterns_01,
    "runtime": Runtime_Patterns,
}
