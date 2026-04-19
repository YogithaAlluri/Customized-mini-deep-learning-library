import numpy as np
from core.tensor import Tensor


class MatMulBackward:
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y

    def backward(self, grad_output):
        """
        out = x @ y
        x: (N, K)
        y: (K, M)
        out: (N, M)
        grad_output: dL/dout, shape (N, M)
        """

        # dL/dx = dL/dout @ y.T
        grad_x = grad_output @ self.y.data.T

        # dL/dy = x.T @ dL/dout
        grad_y = self.x.data.T @ grad_output

        return (grad_x, grad_y)


def matmul(x: Tensor, y: Tensor) -> Tensor:
    """
    Matrix multiplication with autograd support:
    out = x @ y
    """
    out_data = x.data @ y.data
    requires_grad = x.requires_grad or y.requires_grad

    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        out.set_grad_fn(MatMulBackward(x, y), [x, y])

    return out
