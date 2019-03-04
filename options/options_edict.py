"""
# Copyright (c) 2018 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

from easydict import EasyDict as edict

def initialize():
    """
    :return: default settings
    """
    return edict({
        "code_name": None,
        "cover_exist":False,
        "create_path":True,

        "gpu_id": "0",
        "epoch_num": 100,
        "deterministic_train": True,
        "seed": 1,

        "batch_size": 8,
        "learning_rate":1e-4,
        "weight_decay":1e-4,
        "batch_norm":True,
        "finetune":False,

        "curr_epoch": 0,
        "epoches_per_phase": 1,
        "steps_per_epoch": None,

        "model1": None,
        "model2": None,
        "model3": None,
        "model4": None,
        "model5": None,

        
        "loading_threads": 4, # how many cpu core to use to load data, usually 4 is sufficient
        "random_order_load": False,
        "path": None, # the directory contains the datasets
        "extensions": ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"],

        "img_channel": 3,
        "do_imgaug": False,
        "imgaug_order": "default", # or "random"
        # imgaug_order can also be a list, ["affine", "random_crop", "random_zoom", ...]
        # each element represent a process below

        "do_affine": False, # See Documentation of imgaug affine
        "translation_x": (0.0, 0.0),
        "translation_y": (0.0, 0.0),
        "scale_x": (1.0, 1.0),
        "scale_y": (1.0, 1.0),
        "shear": (-0.0, 0.0),
        "rotation": (-0.0, 0.0),
        "aug_bg_color": 255,

        "do_random_crop": False,
        "crop_size": (360, 360),
        "keep_size": True,

        "do_random_zoom": False,
        "pixel_eliminate": (0, 4),
        "sample_independent": False,

        "do_random_flip": False,
        "v_flip_prob": 0.5,
        "h_flip_prob": 0.5,

        "do_random_brightness": False,
        "brightness_vibrator": (1.0, 1.0),
        "multiplier": (1.0, 1.0),
        "multiplier_per_channel": False,
        "linear_contrast": 1.0,

        "do_random_noise": False,
        "gaussian_sigma": (0, 0.1),

        "to_final_size": False,
        "final_size": (224, 224),

        "standardize_size": False,
        "standardize_gcd": 8,

        "img_mean": (0.5, 0.5, 0.5),
        "img_std": (1.0, 1.0, 1.0),
        "img_bias": (0.0, 0.0, 0.0),
    })

if __name__ == "__main__":
    opt = initialize()
    print(opt)