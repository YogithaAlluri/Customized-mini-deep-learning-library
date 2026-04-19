import numpy as np
from core.tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(out_features, in_features),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros((out_features, 1)),
            requires_grad=True
        )

    def forward(self, x: Tensor):
        # CORRECT ORIENTATION:
        # (out_features, in_features) @ (in_features, 1)
        out = self.weight @ x
        out = out + self.bias
        return out

        
        


        


