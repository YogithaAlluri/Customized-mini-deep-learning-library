import numpy as np
from core.grad_function import GradFunction
from core.tensor import Tensor

class MatMul(GradFunction):
    def backward(self, grad_output):
        x, y = self.saved_tensors # x is Weight, y is Input (based on your Linear)

        # dL/dx (Weight) = grad_output @ y.T
        # Shape: (out_features, batch) @ (batch, in_features) -> (out_features, in_features)
        grad_x = grad_output @ y.data.T

        # dL/dy (Input) = x.T @ grad_output
        # Shape: (in_features, out_features) @ (out_features, batch) -> (in_features, batch)
        grad_y = x.data.T @ grad_output

        return grad_x, grad_y

def matmul(x, y):
    # Ensure inputs are Tensors
    if not isinstance(x, Tensor): x = Tensor(x)
    if not isinstance(y, Tensor): y = Tensor(y)
    
    out_data = x.data @ y.data
    requires_grad = x.requires_grad or y.requires_grad

    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        grad_fn = MatMul()
        grad_fn.save_for_backward(x, y)
        out.set_grad_fn(grad_fn, parents=[x, y])

    return out