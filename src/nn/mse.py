import numpy as np
from .module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true      # (4,1)
        sq = diff * diff            # (4,1)

        # Compute mean as a scalar
        mean_value = 1.0 / sq.data.size
        mean_tensor = Tensor(mean_value, requires_grad=False)

        # Reduce to scalar
        loss = (sq * mean_tensor)   # still (4,1)
        return Tensor(np.sum(loss.data), requires_grad=True)



