import torch
import torch.nn as nn
import torch.nn.functional as F

from .signal import SPSignal
from ..utils import shape_numel

class SPGenerator(SPSignal):
    def forward(self, input):
        raise NotImplementedError()
        h0, t0 = input
        return h1, t1, h0, t0

class SPProjectionContext(SPGenerator):
    '''
    '''
    def __init__(self, module, input_shape, output_shape, fixed=False):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.module = module
        if fixed:
            for p in self.module.parameters():
                p.requires_grad = False

    def forward(self, input):
        h0, t0 = input
        t1 = self.module(t0.flatten(1)).view(t0.shape[0:1]+self.output_shape)
        return h0, t1, h0, t0

class SPProjectionContextInput(SPGenerator):
    '''
    '''
    def __init__(self, module_context, module_input, input_shape, output_shape, fixed=False):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.module_context = module_context
        self.module_input = module_input
        if fixed:
            for p in self.module_context.parameters():
                p.requires_grad = False
            for p in self.module_input.parameters():
                p.requires_grad = False

    def forward(self, input):
        h0, t0 = input
        if t0 is not None:
            t1 = self.module_context(t0.flatten(1)).view(t0.shape[0:1]+self.output_shape)
        else:
            t1 = t0
        h1 = self.module_input(h0)
        return h1, t1, h0, t0

class SPFixedDrawContext(SPGenerator):
    '''
    '''
    def __init__(self, places):
        super().__init__()
        self.places = places
        self.numbers = [
            [[1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
            ],
            [[0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
            ],
            [[1,1,1,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,0,0,0],
             [0,0,0,1,1,1,1],
            ],
            [[1,1,1,1,1,1,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
            ],
            [[1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
            ],
            [[1,1,1,1,1,1,1],
             [1,0,0,0,0,0,0],
             [1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
            ],
            [[1,0,0,0,0,0,0],
             [1,0,0,0,0,0,0],
             [1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
            ],
            [[1,1,1,1,1,1,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
            ],
            [[1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
            ],
            [[1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
             [0,0,0,0,0,0,1],
            ],
        ]

    def forward(self, input):
        h0, t0 = input
        numbers = torch.tensor(self.numbers,device=h0.device)
        numbers = numbers.unsqueeze(1).unsqueeze(1)
        numbers = numbers * 2 - 1
        b,c,w,h = h0.shape
        m = int(((w*h)//self.places)**0.5)
        wc = w//m
        hc = h//m
        vizs = []
        for number in t0:
            number = str(int(number))
            viz = torch.zeros((c,w,h),device=h0.device)
            x = 0
            y = 0
            for digit in number:
                d = numbers[int(digit)]
                if (x+m) < w:
                    viz[:,x:x+m,y:y+m] = F.adaptive_avg_pool2d(d.float(), (m,m))
                    x += m
                else:
                    x = 0
                    y += m
            vizs.append(viz.unsqueeze(0))
        t1 = torch.cat(vizs,0)
        #print(vizs)
        return h0, t1, h0, t0

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
