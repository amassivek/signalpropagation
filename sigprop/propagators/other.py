from .propagator import SPPropagator

from .functional_other import forward, fixed, identity

class SPIdentity(SPPropagator):
    '''
    '''
    def __init__(self, module, **kwargs):
        super().__init__()

        self.module = module

    def forward(self, input):
        return identity(input, self.module)

class SPFixed(SPPropagator):
    '''
    '''
    def __init__(self, module, **kwargs):
        super().__init__()

        self.module = module

    def parameters(self):
        return []

    def forward(self, input):
        return fixed(input, self.module)

class SPForward(SPPropagator):
    '''
    '''
    def __init__(self, module, **kwargs):
        super().__init__()

        self.module = module

    def parameters(self):
        return []

    def forward(self, input):
        return forward(input, self.module)

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
