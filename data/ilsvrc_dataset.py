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

# Todo: Not Tested Yet

class ILSVRC_Dataset(Arbitrary_Dataset):
    def __init__(self, args, sources, step_1, step_2, sub_folder, pre_process=None, **options):
        super().__init__(args, sources, step_1, step_2, sub_folder, pre_process, options)
        
    def prepare(self):
        self.dataset = self.load_dataset()
    
    def load_dataset(self):
        """
        :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
        """
        dataset = {}
        path = os.path.expanduser(self.args.path)
        classes = os.listdir(path)
        classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
        for i, cls in enumerate(classes):
            if self.verbose:
                print('Loading {}th {} class.'.format(i, cls))
            dataset.update({cls: [_ for _ in os.listdir(os.path.join(path, cls)) if
                                  misc.extension_check(_, self.args.extensions)]})
        print("Dataset loading is complete.")
        return dataset