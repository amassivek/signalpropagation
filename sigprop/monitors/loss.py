import torch
import torch.nn.functional as F

from .monitor import SPMonitor

class SPLoss(SPMonitor):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def reset(self):
        raise NotImplementedError()

    def metrics(self):
        raise NotImplementedError()

    def __call__(self,sp_learn,h1,t1,h0,t0,context):
        loss = self.loss(sp_learn,h1,t1,h0,t0,context)
        raise NotImplementedError()

class SPLabelMetrics(SPLoss):
    def __init__(self, name, loss):
        super().__init__(loss)

        self.name = name
        self.reset()

    def reset(self):
        self.loss_sum = 0.
        self.acc_sum = 0.
        self.count = 0.
        self.acc_last = 0.
        self.loss_last = 0.

    def metrics(self):
        info = "[{}] Acc: {:.4f} ({:.4f}, {}/{}) \t Loss: {:.4f} ({:.4f})".format(
            self.name,
            self.acc_last,
            self.acc_sum / (self.count+1e-6),
            int(self.acc_sum), int(self.count),
            self.loss_last,
            self.loss_sum / (self.count+1e-6)
        )
        return info

    def __call__(self,sp_learn,h1,t1,h0,t0,context):
        loss = self.loss(sp_learn,h1,t1,h0,t0,context)
        #print(h1.shape,t1.shape,h0.shape,t0.shape,context.shape)
        with torch.no_grad():
            self.loss_sum += loss.item() * h1.shape[0]
            acc_mask = (h1.flatten(1) @ t1.flatten(1).t())
            acc_mask = acc_mask.argmax(1)
            if h1.shape[0] != t1.shape[0]:
                acc_mask = acc_mask == context
            else:
                yy = context
                yy = yy.unsqueeze(1) == yy.unsqueeze(0)
                yy = yy.float() * 2 - 1
                acc_mask = F.one_hot(acc_mask, t1.shape[0]).float()
                acc_mask = (acc_mask == yy).any(1)
            self.acc_sum += acc_mask.sum().item()
            self.count += h1.shape[0]
            self.loss_last = loss.item()
            self.acc_last = acc_mask.sum().item() / h1.shape[0]
        return loss

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
