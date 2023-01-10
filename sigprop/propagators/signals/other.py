from ..propagator import SPPropagator

class SPFixed(SPPropagator):
    '''
    '''
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, input):
        h0, t0, context = input

        h0 = h0.detach()

        if t0 is not None:
            t0 = t0.detach()
            h1,t1,h0,t0 = self.module((h0, t0))

            t1 = t1.detach()

        else:
            h1,t1,h0,t0 = self.module((h0, t0))

        h1 = h1.detach()

        return (h1, t1)

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
