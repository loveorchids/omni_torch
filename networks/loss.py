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

import torch
import torch.nn as nn
    
class MSE_Loss(nn.Module):
    def __init__(self, sum_dim=None, sqrt=False, dimension_warn=0):
        super().__init__()
        self.sum_dim = sum_dim
        self.sqrt = sqrt
        self.dimension_warn = dimension_warn
    
    def forward(self, x, y):
        assert x.shape == y.shape
        if self.sum_dim:
            mse_loss = torch.sum((x - y) ** 2, dim=self.sum_dim)
        else:
            mse_loss = torch.sum((x - y) ** 2)
        if self.sqrt:
            mse_loss = torch.sqrt(mse_loss)
        mse_loss = torch.sum(mse_loss) / mse_loss.nelement()
        if len(mse_loss.shape) > self.dimension_warn:
            raise ValueError("The shape of mse loss should be a scalar, but you can skip this"
                             "error by change the dimension_warn explicitly.")
        return mse_loss
    
class KL_Divergence(nn.Module):
    def __init__(self, sum_dim=None, sqrt=False, dimension_warn=0):
        super().__init__()
        self.sum_dim = sum_dim
        self.sqrt = sqrt
        self.dimension_warn = dimension_warn
        
    def forward(self, x, y):
        # Normalize
        x = x.view(x.size(0), x.size(1), -1)
        x = x / x.norm(1, dim=-1).unsqueeze(-1)
        y = y.view(y.size(0), y.size(1), -1)
        y = y / y.norm(1, dim=-1).unsqueeze(-1)
        loss = torch.sum((y * (y.log() - x.log())), dim=self.sum_dim)
        return loss.squeeze()

class Triplet_Loss(nn.Module):
    def __init__(self, mode, margin=0.2, engine=MSE_Loss, weighted=False,
                 sum_dim=None, sqrt=False):
        """
        :param mode: minus and divide mode
        :param margin: a mininum distance to keep the triplet distance discriminative
        :param weighted: amplify the wrong case more
        """
        super().__init__()
        self.mode = mode
        self.margin = margin
        self.weighted = weighted
        self.mse = engine(sum_dim, sqrt, dimension_warn=1)
    
    def forward(self, positive, anchor, negative):
        if self.mode == "minus":
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
    
class KL_Triplet_Loss(nn.Module):
    def __init__(self, symmetric=True):
        """
        :param symmetric: if symmetric, we will use JS Divergence, if not KL Divergence will be used.
        """
        super().__init__()
        self.symmetric = symmetric
        self.engine = nn.KLDivLoss()
        
    def forward(self, x, y):
        if len(x.shape)==4 and len(y.shape)==4:
            x = x.view(x.size(0) * x.size(1), -1)
            y = y.view(y.size(0) * y.size(1), -1)
        elif len(x.shape)==2 and len(y.shape)==2:
            pass
        else:
            raise TypeError("We need a tensor of either rank 2 or rank 4.")
        if self.symmetric:
            loss = self.engine(x, y)
        else:
            loss = self.engine(x, y) + self.engine(y, x)
        return loss
    
if __name__ == "__main__":
    klTriplket = KL_Triplet_Loss()
    x = torch.randn(64, 256, 10, 10)
    y = torch.randn(64, 256, 10, 10)
    loss = klTriplket(x, y)
    print(loss.shape, loss)
    