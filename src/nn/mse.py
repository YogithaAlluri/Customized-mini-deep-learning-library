import numpy as np
from .module import Module
from core.tensor import Tensor


class MSEBackward:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_pred.data.size  # number of elements

    def backward(self, grad_output):
        # grad_output is a scalar (dL/dLoss), usually 1.0
        diff = self.y_pred.data - self.y_true.data  # NumPy array

        # dL/dy_pred = 2 * (y_pred - y_true) / n * grad_output
        grad_y_pred = (2.0 / self.n) * diff * grad_output

        # dL/dy_true = -dL/dy_pred
        grad_y_true = -grad_y_pred

        return (grad_y_pred, grad_y_true)


class MSELoss(Module):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Compute scalar loss value with NumPy
        diff = y_pred.data - y_true.data
        sq = diff * diff
        loss_value = np.mean(sq)  # plain float

        # Wrap as Tensor that requires grad
        loss = Tensor(loss_value, requires_grad=True)

        # Attach custom grad_fn so autograd knows how to backprop
        loss.set_grad_fn(MSEBackward(y_pred, y_true), [y_pred, y_true])

        return loss














