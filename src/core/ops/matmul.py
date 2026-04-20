import numpy as np
from core.grad_function import GradFunction
from core.tensor import Tensor

class MatMul(GradFunction):
    def backward(self, grad_output):
        x, y = self.saved_tensors

        # dL/dx = grad_output @ y.T
        grad_x = grad_output @ y.data.T

        # dL/dy = x.data.T @ grad_output
        grad_y = x.data.T @ grad_output

        return grad_x, grad_y


def matmul(x, y):
    out_data = x.data @ y.data
    requires_grad = x.requires_grad or y.requires_grad

    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        grad_fn = MatMul()
        grad_fn.save_for_backward(x, y)
        out.set_grad_fn(grad_fn, parents=[x, y])

    return out

