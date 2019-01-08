from easydict import EasyDict as edict

BaseOptions=edict({
    "code_name": None,
    "cover_exist":False,
    "create_path":True,

    "gpu_id":"0",
    "epoch_num":2000,
    "deterministic_train": True,
    "seed": 88,

    "batch_size": 8,
    "learning_rate":1e-4,
    "weight_decay":1e-4,
    "batch_norm":True,
    "finetune":False,

    "model1": None,
    "model2": None,
    "model3": None,
    "model4": None,
    "model5": None,

    "loading_threads": 2,
    "random_order_load": False,
    "path": None,
    "extensions": ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"]
})

ImageAugmentation=edict({
    "img_channel": 3,
    "imgaug_engine": "cv2",

    "do_imgaug": True,
    "imgaug_order": "default", # or "random"
    # imgaug_order can also be a list, ["affine", "random_crop", "random_zoom", ...]
    # each element represent a process below

    "do_affine": False,
    "translation": (0.0, 0.0),
    "scale": (1.0, 1.0),
    "shear": (-0.0, 0.0),
    "rotation": (-0.0, 0.0),
    "aug_bg_color": (-0.0, 0.0),

    "do_random_crop": False,
    "crop_size": (360, 360),
    "keep_size": True,

    "do_random_zoom": False,
    "pixel_eliminate": (0, 50),
    "sample_independent": False,

    "do_random_flip": False,
    "v_flip_prob": 0.5,
    "h_flip_prob": 0.5,

    "do_random_brightness": False,
    "brightness_vibrator": (1.0, 1.0),
    "multiplier": (1.0, 1.0),
    "per_channel_multiplier": 0.2,
    "linear_contrast": 1.0,

    "do_random_noise": False,
    "gaussian_sigma": (0, 0.1),

    "to_final_size": False,
    "final_size": (224, 224),

    "standardize_size": False,
    "standardize_gcd": 8,

    "img_mean": (0.5, 0.5, 0.5),
    "img_std": (1.0, 1.0, 1.0),
})