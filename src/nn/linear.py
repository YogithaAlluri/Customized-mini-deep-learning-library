from .module import Module
from src.core.parameter import Parameter

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(None)
        self.bias = Parameter(None)

    def forward(self, x):
        pass

