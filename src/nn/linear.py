import numpy as np
from core.tensor import Tensor
from core.parameter import Parameter
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(
            np.random.randn(out_features, in_features)
        )
        self.bias = Parameter(
            np.zeros((out_features, 1))
        )

    def forward(self, x: Tensor):
        out = self.weight @ x
        out = out + self.bias
        return out
