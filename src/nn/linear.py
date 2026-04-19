import numpy as np
from core.parameter import Parameter
from .module import Module


class Linear(Module):
    """
    A fully connected linear layer: y = xW^T + b
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        # Initialize weights with small random values
        self.weight = Parameter(
            np.random.randn(out_features, in_features) * 0.01
        )

        # Initialize bias with zeros
        self.bias = Parameter(np.zeros((out_features, 1)))
            
        

    def forward(self, x):
        out = x @ self.weight.T
        out = out + self.bias
        return out
        
        


        


