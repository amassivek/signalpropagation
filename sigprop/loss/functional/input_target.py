import torch
import torch.nn.functional as F

from .utils import *

def v9_input_target_max_all(sp_learn,h1,t1,h0,t0,context):

    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        yy = t0 @ t0.t()
    else:
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yym = yy.max(1,keepdim=True)[0]
    yy = (yy == yym).float()

    l = soft_target_cross_entropy(y, yy)

    return l

def v14_input_target_max_rand(sp_learn,h1,t1,h0,t0,context):

    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        yy = t0 @ t0.t()
    else:
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    l  = F.cross_entropy(y, yy.argmax(1))

    return l

class v15_input_target_topk(object):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def __call__(self,sp_learn,h1,t1,h0,t0,context):

        if len(t0.shape) == 2:
            t0 = t0.flatten(1)
            yy = t0 @ t0.t()
        else:
            t0 = t0.flatten(2).permute(2,0,1)
            yy = (t0 @ t0.permute(0,2,1)).mean(0)

        if len(t1.shape) > 2:
            h1 = h1.flatten(2).permute(2,0,1)
            t1 = t1.flatten(2).permute(2,1,0)
            y = (h1 @ t1).mean(0)

        else:
            h1 = h1.flatten(1)
            t1 = t1.flatten(1)

            y = h1 @ t1.t()

        yym = yy.topk(self.topk,dim=1)[1]
        yy = F.one_hot(yym,yy.shape[1]).sum(1).clamp(0,1).float()

        l = soft_target_cross_entropy(y, yy)

        return l
