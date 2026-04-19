from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        # y_pred, y_true: Tensor of shape (N, 1)

        # 1. Tensor operations only
        diff = y_pred - y_true          # Tensor
        sq = diff * diff                # Tensor

        # 2. Mean factor as a Tensor (no grad needed)
        n = sq.data.size                # plain Python int
        mean = Tensor(1.0 / n, requires_grad=False)

        # 3. Elementwise mean
        loss = sq * mean                # still Tensor, same shape

        # 4. Reduce to scalar using a simple sum in Python,
        #    but KEEP the Tensor graph by summing Tensor elements.
        total = None
        for i in range(loss.data.shape[0]):
            # each row is a Tensor via broadcasting
            row_value = loss.data[i, 0]             # scalar value
            row_tensor = Tensor(row_value, requires_grad=True)
            total = row_tensor if total is None else total + row_tensor

        return total













