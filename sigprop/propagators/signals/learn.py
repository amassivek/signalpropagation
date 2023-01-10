from ..propagator import SPPropagator

class SPLearn(SPPropagator):
    '''
    '''
    def __init__(self, module):
        super().__init__()

        self.module = module

    def parameters(self):
        return []

    def train_(self,h1,t1,h0,t0,context):
        raise NotImplementedError()

    def eval_(self,h1,t1,h0,t0,context):
        raise NotImplementedError()

    def forward(self, input):
        h0, t0, context = input

        h0 = h0.detach()

        if t0 is not None:
            t0 = t0.detach()
            h1,t1,h0,t0 = self.module((h0, t0))

            context = context.detach()
            if self.training:
                loss, correct, total = self.train_(h1,t1,h0,t0,context)
            else:
                loss, correct, total = self.eval_(h1,t1,h0,t0,context)

            t1 = t1.detach()

        else:
            h1,t1,h0,t0 = self.module((h0, t0))

        h1 = h1.detach()

        return (h1, t1)

class SPLoss(SPLearn):
    def __init__(self, module, optimizer, loss):
        super().__init__(module)
        self.optimizer = optimizer
        self.loss = loss

    def train_(self,h1,t1,h0,t0,context):
        loss = self.loss(self,h1,t1,h0,t0,context)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item(), None, None

    def eval_(self,h1,t1,h0,t0,context):
        loss = self.loss(self,h1,t1,h0,t0,context)
        return loss.item(), None, None

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
