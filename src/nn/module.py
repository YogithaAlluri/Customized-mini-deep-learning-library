class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self._parameters.values()

    def add_module(self, name, module):
        self._modules[name] = module

