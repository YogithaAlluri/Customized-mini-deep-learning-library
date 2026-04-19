import numpy as np
from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true      # Tensor subtraction
        sq = diff * diff            # Tensor multiplication

        # Compute mean using Tensor ops
        mean_factor = Tensor(1.0 / sq.data.size, requires_grad=False)
        loss = sq * mean_factor     # Tensor * Tensor keeps graph

        return loss


