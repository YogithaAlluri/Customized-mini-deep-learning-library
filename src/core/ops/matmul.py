import numpy as np
from core.grad_function import GradFunction
from core.tensor import Tensor


class MatMul(GradFunction):
    """
    Gradient function for matrix multiplication: z = x @ y
    """

    def backward(self, grad_output):
        x, y = self.saved_tensors

        # dz/dx = grad_output @ y.T
        grad_x = grad_output @ y.data.T

        # dz/dy = x.T @ grad_output
        grad_y = x.data.T @ grad_output

        return grad_x, grad_y


def matmul(x, y):
    """
    Perform x @ y with autograd support.
    """
    # Forward pass
    out_data = x.data @ y.data

    # Create output tensor
    out = Tensor(out_data, requires_grad=(x.requires_grad or y.requires_grad))

    # Attach grad function
    if out.requires_grad:
        grad_fn = MatMul()
        grad_fn.save_for_backward(x, y)
        out.set_grad_fn(grad_fn, parents=[x, y])

    return out
