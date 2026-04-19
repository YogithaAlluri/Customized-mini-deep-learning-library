import numpy as np
from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        sq = diff * diff

        # Mean factor
        mean = Tensor(1.0 / sq.data.size, requires_grad=False)
        reduced = sq * mean  # (N,1)

        # ---- Reduce to scalar WITHOUT breaking graph ----
        # We multiply each element by a Tensor(1) to keep graph
        total = None
        for i in range(reduced.data.shape[0]):
            elem_value = reduced.data[i, 0]
            elem = Tensor(elem_value, requires_grad=True)
            total = elem if total is None else total + elem

        return total





