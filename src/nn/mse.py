from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        sq = diff * diff

        mean = Tensor(1.0 / sq.data.size, requires_grad=False)
        reduced = sq * mean  # (N,1)

        # Proper reduction: multiply by Tensor(1) to keep graph
        total = Tensor(0.0, requires_grad=True)

        # Loop through rows
        for i in range(reduced.data.shape[0]):
            # Create a mask row: 1 at row i, 0 elsewhere
            mask_data = (reduced.data * 0)
            mask_data[i, 0] = 1.0
            mask = Tensor(mask_data, requires_grad=False)

            elem = reduced * mask   # picks out row i as a Tensor
            total = total + elem    # accumulate using Tensor ops

        return total






