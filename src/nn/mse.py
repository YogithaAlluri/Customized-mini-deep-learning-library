from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true      # (N,1)
        sq = diff * diff            # (N,1)

        # Mean factor (scalar Tensor)
        mean = Tensor(1.0 / sq.data.size, requires_grad=False)
        reduced = sq * mean         # (N,1)

        # ---- Reduce to scalar using Tensor addition ----
        total = Tensor(0.0, requires_grad=True)

        # Add each element using Tensor ops (NO numpy!)
        for i in range(reduced.data.shape[0]):
            # isolate row i using multiplication
            row_mask = Tensor((reduced.data * 0), requires_grad=False)
            row_mask.data[i, 0] = 1.0

            elem = reduced * row_mask   # picks out row i
            total = total + elem        # accumulate

        return total











