import numpy as np
from core.tensor import Tensor
from .module import Module

class MSEBackward:
    def __init__(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true
    def backward(self, grad_output):
        n = self.y_pred.data.size
        grad_y_pred = (2.0 / n) * (self.y_pred.data - self.y_true.data) * grad_output
        return (grad_y_pred, -grad_y_pred)

class MSELoss(Module):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        loss_val = np.mean((y_pred.data - y_true.data)**2)
        out = Tensor(loss_val, requires_grad=True)
        out.set_grad_fn(MSEBackward(y_pred, y_true), [y_pred, y_true])
        return out