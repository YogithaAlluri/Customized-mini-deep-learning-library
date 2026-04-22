import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from core.tensor import Tensor
from nn.linear import Linear
from nn.relu import ReLU
from nn.mse import MSELoss
from optim.sgd import SGD


class FullModel:
    def __init__(self):
        self.l1 = Linear(1, 4)
        self.relu = ReLU()
        self.l2 = Linear(4, 1)

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def __call__(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def test_training_full_model():
    np.random.seed(42)

    model = FullModel()
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Training data: y = 2x + 3
    # Shape convention: x = (in_features, batch_size)
    x_data = np.array([[1.0, 2.0, 3.0, 4.0]])
    y_data = np.array([[5.0, 7.0, 9.0, 11.0]])

    x = Tensor(x_data, requires_grad=True)
    y = Tensor(y_data, requires_grad=False)

    losses = []

    for epoch in range(200):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data}")

    print("\nInitial loss:", losses[0])
    print("Final loss:", losses[-1])
    print("\nPredictions after training:\n", pred.data)
    print("\nTargets:\n", y.data)
    print("\nL1 weight:\n", model.l1.weight.data)
    print("L1 bias:\n", model.l1.bias.data)
    print("\nL2 weight:\n", model.l2.weight.data)
    print("L2 bias:\n", model.l2.bias.data)

    assert losses[-1] < losses[0], "Loss did not decrease during training"

    print("\ntest_training_full_model passed")


if __name__ == "__main__":
    test_training_full_model()
    print("\nComplete training test passed successfully.")