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
        self.bias = Parameter(
            np.zeros(out_features)
        )

    def forward(self, x):
        """
        Compute the linear transformation.
        x: Tensor of shape (batch_size, in_features)
        """
        # x @ W^T
        out = x.data @ self.weight.data.T

        # Add bias
        out = out + self.bias.data

        # Wrap output in a Tensor (no autograd yet)
        from core.tensor import Tensor
        return Tensor(out, requires_grad=x.requires_grad)


