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
            # Multiply by a mask row to isolate element i
            mask_data = (reduced.data * 0)
            mask_data[i, 0] = 1.0
            mask = Tensor(mask_data, requires_grad=False)

            elem = reduced * mask   # picks out row i as a Tensor
            total = total + elem    # accumulate

        return total










