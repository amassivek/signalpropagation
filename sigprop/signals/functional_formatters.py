import torch
import torch.nn.functional as F

def label_number_to_onehot(t0, num_classes):
    if t0 is not None:
        if num_classes is None:
            num_classes = int(t0.max())
        t0 = F.one_hot(t0, num_classes).float()
    return t0

def distinct_batch_labels_onehot(t0):
    if t0 is not None:
        t0 = F.one_hot(
            torch.arange(t0.shape[1], device=t0.device),
            t0.shape[1]
        ).float()
    return t0

def distinct_batch_labels(t0):
    if t0 is not None:
        t0 = torch.arange(t0.shape[1], device=t0.device).float()
    return t0

def label_vector_to_number(t0):
    if t0 is not None:
        t0 = t0.argmax(1,keepdim=True).float()
    return t0
