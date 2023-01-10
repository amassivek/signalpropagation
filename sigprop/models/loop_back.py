import torch

from .model import SPModel

class SPLoopBack(SPModel):
    def __init__(self, model, signal, error):
        super().__init__(model, signal)

        self.error = error

    def forward(self, input):
        x, context = input
        with torch.no_grad():
            h0, _ = self.signal((x, None, None))
            h1, _, _ = self.model((h0, None, None))
        if context is not None:
            with torch.no_grad():
                e = self.error(h1, context)
            h0, t0 = self.signal((x, e, context))
            h1, t1, context = self.model((h0, t0, context))
        return h1

class SPForwardLoopBack(SPModel):
    def __init__(self, model, signal, error):
        super().__init__(model, signal)

        self.error = error

    def forward(self, input):
        x, context = input
        h0, t0 = self.signal((x, context, context))
        h1, t1, _ = self.model((h0, t0, context))
        if context is not None:
            with torch.no_grad():
                e = self.error(h1, context)
            h0, t0 = self.signal((x, e, context))
            h1, t1, context = self.model((h0, t0, context))
        return h1

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
