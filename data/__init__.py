import torch
import numpy as np
import scipy.io as sio

from data.ilsvrc import ILSVRC
from data.img2img import Img2Img
from data.arbitrary import Arbitrary

class Data:
    def __getitem__(self, idx):
        result = [_[idx] for _ in self.dataset.values()]
        return result