import numpy as np
from .module import Module
from core.tensor import Tensor


class ReLU(Module):
    """
    ReLU activation: max(0, x)
    """

    def forward(self, x):
        # Apply ReLU element-wise
        out = np.maximum(0, x.data)

        # Wrap output in a Tensor
        return Tensor(out, requires_grad=x.requires_grad)

