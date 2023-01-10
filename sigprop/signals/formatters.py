import torch
import torch.nn.functional as F

from .signal import SPSignal

from .functional_formatters import *

class SPFormatter(SPSignal):
    def __init__(self):
        super().__init__()

class SPLabelNumberToOnehot(SPFormatter):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, input):
        h0, t0 = input
        t1 = label_number_to_onehot(t0, self.num_classes)
        return h0, t1

class SPDistinctBatchLabelsOnehot(SPFormatter):
    def forward(self, input):
        h0, t0 = input
        t1 = distinct_batch_labels_onehot(t0)
        return h0, t1

class SPDistinctBatchLabels(SPFormatter):
    def forward(self, input):
        h0, t0 = input
        t1 = distinct_batch_labels(t0)
        return h0, t1

class SPLabelVectorToNumber(SPFormatter):
    def forward(self, input):
        h0, t0 = input
        t1 = label_vector_to_number(t0)
        return h0, t1

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
