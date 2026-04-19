import numpy as np
from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        # Tensor operations (keeps graph)
        diff = y_pred - y_true          # (N,1)
        sq = diff * diff                # (N,1)

        # Mean factor (scalar)
        mean = Tensor(1.0 / sq.data.size, requires_grad=False)

        # Elementwise mean
        reduced = sq * mean             # (N,1)

        # ---- Reduce to scalar WITHOUT breaking graph ----
        # We manually sum elements using Tensor addition
        total = None
        flat = reduced.data.flatten()

        for i in range(len(flat)):
            elem = Tensor(flat[i], requires_grad=True)
            total = elem if total is None else total + elem

        return total




