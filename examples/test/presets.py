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

def GeneralPattern_01(args):
    args.path = "~/Pictures/dataset/cifar-10"
    args.random_order_load = False
    args.loading_threads = 4
    args.img_channel = 3
    args.curr_epoch = 0
    args.do_imgaug = False

    args.img_mean = (0.0, 0.0, 0.0)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args

def UniquePattern_01(args):
    args.adam_epsilon = 1e-8
    return args

def RuntimePattern(args):
    args.cover_exist = True
    args.code_name = "cifar_torch"
    args.finetune = False

    args.deterministic_train = False
    args.epoch_num = 200
    args.batch_size = 256

    args.learning_rate = 1e-4
    args.weight_decay = 1e-6
    args.gpu_id = "0"
    return args


PRESET = {
    "general": GeneralPattern_01,
    "unique": UniquePattern_01,
    "runtime": RuntimePattern,
}