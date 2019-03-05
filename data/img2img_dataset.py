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

import os
import omni_torch.data.misc as misc
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
import omni_torch.data as data

ALLOW_WARNING = data.ALLOW_WARNING

class Img2Img_Dataset(Arbitrary_Dataset):
    def __init__(self, args, sources, step_1, step_2, auxiliary_info, pre_process=None,
                 one_to_one=True, **options):
        assert len(sources) is 2, "In img2img dataset only two sources are allowed."
        assert len(step_1) is 2, "In img2img dataset only two sources are allowed."
        assert len(step_2) is 2, "In img2img dataset only two sources are allowed."
        super().__init__(args, sources, step_1, step_2, auxiliary_info, pre_process, **options)
        self.one_to_one = one_to_one
        
    def prepare(self):
        self.dataset = self.load_dataset()

    def load_dataset(self):
        A_B_folder = self.sources
        dataset = []
        path = os.path.expanduser(self.args.path)
        assert len(A_B_folder) == 2, "A_B_folder should be the name of source and target folder."
        source = os.path.join(path, A_B_folder[0])
        target = os.path.join(path, A_B_folder[1])
        assert os.path.isdir(source) and os.path.isdir(target), "one of the folder does not exist."
        source_imgs = [os.path.join(path, A_B_folder[0], _) for _ in os.listdir(source)
                       if misc.extension_check(_, self.args.extensions)]
        target_imgs = [os.path.join(path, A_B_folder[1], _) for _ in os.listdir(target)
                       if misc.extension_check(_, self.args.extensions)]
        if self.one_to_one:
            assert len(source_imgs) == len(target_imgs)
            if self.verbose: print("Sorting files...")
            source_imgs.sort()
            target_imgs.sort()
        else:
            dim = max(len(source_imgs), len(target_imgs))
            if len(source_imgs) < len(target_imgs):
                source_imgs = misc.compliment_dim(source_imgs, dim)
            else:
                target_imgs = misc.compliment_dim(target_imgs, dim)
        for i in range(len(source_imgs)):
            if self.verbose and i % 100 == 0:
                print("{} samples has been loaded...".format(i))
            dataset.append([source_imgs[i], target_imgs[i]])
        print('Dataset loading is complete.')
        return dataset