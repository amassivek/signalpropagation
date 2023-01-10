from ..module import SPModule

class SPModel(SPModule):
    def __init__(self, model, signal):
        super().__init__()

        self.signal = signal
        self.model = model

    def parameters(self):
        params = []
        for m in self.model.modules():
            if isinstance(m, SPModule):
                params.extend(list(m.parameters()))
        return params

    @classmethod
    def manage(cls, manager, module, **kwords):
        return manager.set_model(module, cls, **kwords)

class SPForward(SPModel):
    def forward(self, input):
        x, context = input
        h0, t0 = self.signal((x, context, context))
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
