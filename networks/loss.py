"""
# Copyright (c) 2019 Wang Hanqin
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

import torch
import torch.nn as nn


class Triplet_Loss(nn.Module):
    def __init__(self, metric="cosine", margin=0.2, weighted=False, sum_dim=None, sqrt=False):
        """
        :param mode: minus and divide mode
        :param margin: a mininum distance to keep the triplet distance discriminative
        :param weighted: amplify the wrong case more
        """
        super().__init__()
        self.margin = margin
        self.weighted = weighted
        self.metric = metric
    
    def forward(self, positive, anchor, negative):
        if self.metric.lower() in ["l2", "euclidean", "l_2"]:
            dist = self.mse(anchor, positive) - self.mse(anchor, negative) + self.margin
        elif self.mode == "divide":
            dist = self.mse(anchor, positive) + 1 / (self.mse(anchor, negative) + self.margin)
        else:
            raise NotImplementedError
        if self.weighted:
            weight = dist / dist.norm(1)
            dist = torch.sum(dist * weight) / dist.nelement()
        else:
            dist = torch.sum(dist) / dist.nelement()
        while len(dist.shape) > 0:
            print("actual loss shape is larger than zero, sum is somewhat wrong. fixing it...")
            dist = torch.sum(dist)
        return dist
    
class JS_Divergence(nn.Module):
    def __init__(self):
        super().__init__()
        self.engine = nn.KLDivLoss()
        
    def forward(self, x, y):
        return self.engine(x, y) + self.engine(y, x)

    