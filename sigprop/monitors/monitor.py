class SPMonitor(object):
    pass

class SPGroup(SPMonitor):
    def __init__(self):
        super().__init__()

        self.monitors = []

    def add(self, monitor):
        self.monitors.append(monitor)

    def reset(self):
        for m in self.monitors:
            m.reset()

    def metrics(self):
        stats = []
        for i,m in enumerate(self.monitors):
            stats.append(m.metrics())
        return "\n".join(stats)

__all__ = []
g = globals()
for name, obj in g.copy().items():
    if name.startswith("SP"):
        new_name = name[2:]
        g[new_name] = obj
        __all__.append(new_name)
del g
