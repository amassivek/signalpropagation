import torch
import torch.nn.functional as F

from .utils import *

def v1_input_label_direct(sp_learn,h1,t1,h0,t0,context):
    if len(t1.shape) > 2 and len(t0.shape) == 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    elif len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    l  = F.cross_entropy(y, context)

    return l
