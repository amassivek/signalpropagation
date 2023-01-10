from torch import nn

class SPManager(object):
    '''
    '''
    def __init__(self):
        super().__init__()

        self.model = None
        self.signal = None
        self.propagators = []

    def set_signal(self, module, **kwords):
        pass

    def add_propagator(self, module, propagator_cls=None, **kwords):
        pass

    def set_model(self, module, **kwords):
        pass

class SPPrepared(SPManager):
    '''
    '''
    def __init__(self, model_cls, propagator_cls, signal_cls):
        super().__init__()

        self.propagator_cls = propagator_cls
        self.model_cls = model_cls
        self.signal_cls = signal_cls

        self.model = None
        self.signal = None
        self.propagators = []

    def set_signal(self, module, **kwords):
        if self.signal is not None:
            raise RuntimeError(
                "Signal already set."
            )
        self.signal = self.signal_cls(module, **kwords)
        #self.propagators.append(self.signal)
        return self.signal

    def add_propagator(self, module, propagator_cls=None, **kwords):
        if propagator_cls is None:
            propagator_cls = self.propagator_cls
        propagator = propagator_cls(module, **kwords)
        self.propagators.append(propagator)
        return propagator

    def set_model(self, module, **kwords):
        if self.model is not None:
            raise RuntimeError(
                "Model already set."
            )
        if self.signal is None:
            raise RuntimeError(
                "set_signal() before set_model()."
            )
        self.model = self.model_cls(module, self.signal, **kwords)
        return self.model

class SPPreset(SPPrepared):
    '''
    '''
    def __init__(self,
            model_cls, propagator_cls, signal_cls,
            optimizer_builder,
            model_kwargs=None, propagator_kwargs=None, signal_kwargs=None,
        ):
        super().__init__(model_cls, propagator_cls, signal_cls)

        self.optimizer_builder = optimizer_builder
        if model_kwargs is None:
            self.model_kwords = dict()
        else:
            self.model_kwords = model_kwargs
        if propagator_kwargs is None:
            self.propagator_kwords = dict()
        else:
            self.propagator_kwords = propagator_kwargs
        if signal_kwargs is None:
            self.signal_kwords = dict()
        else:
            self.signal_kwords = signal_kwargs

    def config_signal(self, **kwords):
        if self.signal is not None:
            raise RuntimeError(
                "config_signal() before set_signal();"
                " signal already set."
            )
        self.signal_kwords.update(kwords)

    def config_propagator(self, **kwords):
        if len(self.propagators) > 0:
            raise RuntimeError(
                "config_propagator() before first call to add_propagator();"
                " at least one propagator already added."
            )
        self.propagator_kwords.update(kwords)

    def config_model(self, **kwords):
        if self.model is not None:
            raise RuntimeError(
                "config_model() before set_model();"
                " model already set."
            )
        self.model_kwords.update(kwords)

    def set_signal(self, module, **kwords):
        if "optimizer" not in kwords:
            kwords["optimizer"] = self.optimizer_builder(module)
        kwords_all = self.signal_kwords.copy()
        kwords_all.update(kwords)
        return super().set_signal(module, **kwords_all)

    def add_propagator(self, module, propagator_cls=None, **kwords):
        if "optimizer" not in kwords:
            kwords["optimizer"] = self.optimizer_builder(module)
        kwords_all = self.propagator_kwords.copy()
        kwords_all.update(kwords)
        return super().add_propagator(module, propagator_cls, **kwords_all)

    def set_model(self, module, **kwords):
        kwords_all = self.model_kwords.copy()
        kwords_all.update(kwords)
        return super().set_model(module, **kwords_all)

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
