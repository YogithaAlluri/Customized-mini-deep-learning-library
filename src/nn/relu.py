import numpy as np
from .module import Module
from core.tensor import Tensor


class ReLUBackward:
    def __init__(self, mask):
        # mask is a NumPy array of 0/1 with same shape as x.data
        self.mask = mask

    def backward(self, grad_output):
        # grad_output is a NumPy array (same shape as mask)
        # dL/dx = dL/dout * 1{x > 0}
        return (grad_output * self.mask,)


class ReLU(Module):
    """
    ReLU activation: max(0, x)
    """

    def forward(self, x: Tensor) -> Tensor:
        # forward on raw data
        out_data = np.maximum(0.0, x.data)

        # wrap in Tensor
        out = Tensor(out_data, requires_grad=x.requires_grad)

        if x.requires_grad:
            # mask for backward: 1 where x > 0, else 0
            mask = (x.data > 0).astype(float)
            # connect to graph
            out.set_grad_fn(ReLUBackward(mask), [x])

        return out


