from ..module import SPModule

class SPPropagator(SPModule):
    @classmethod
    def manage(cls, manager, module, **kwords):
        return manager.add_propagator(module, cls, **kwords)
