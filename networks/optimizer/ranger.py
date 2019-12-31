import math
import torch
from torch.optim.optimizer import Optimizer
import itertools as it
from .lookahead import Lookahead
from .radam import RAdam
from .ralamb import Ralamb

def Ranger(params, alpha=0.5, k=6, *args, **kwargs):
     radam = RAdam(params, *args, **kwargs)
     return Lookahead(radam, alpha, k)

def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)