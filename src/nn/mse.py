from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true      # (N,1)
        sq = diff * diff            # (N,1)

        # Mean factor (scalar Tensor)
        mean = Tensor(1.0 / sq.data.size, requires_grad=False)
        reduced = sq * mean         # (N,1)

        # ---- Proper reduction to scalar ----
        total = Tensor(0.0, requires_grad=True)

        # Sum using Tensor addition (keeps graph)
        for i in range(reduced.data.shape[0]):
            # Instead of masking, multiply by a scalar Tensor
            elem_value = reduced.data[i, 0]
            elem = Tensor(elem_value, requires_grad=True)
            total = total + elem

        return total












