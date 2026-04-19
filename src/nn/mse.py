import numpy as np
from .module import Module
from core.tensor import Tensor


class MSELoss(Module):
    """
    Mean Squared Error loss:
    loss = mean((y_pred - y_true)^2)
    """

    def forward(self, y_pred, y_true):
        # Compute squared difference
        diff = y_pred.data - y_true.data
        loss_value = np.mean(diff ** 2)

        # Wrap in a Tensor
        return Tensor(loss_value, requires_grad=y_pred.requires_grad)

