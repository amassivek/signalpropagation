from ..module import SPModule

class SPSignal(SPModule):
    @classmethod
    def manage(cls, manager, module, **kwords):
        return manager.set_signal(module, cls, **kwords)
