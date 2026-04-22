import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from core.tensor import Tensor
from nn.linear import Linear
from nn.relu import ReLU
from nn.mse import MSELoss
from optim.sgd import SGD


class SimpleModel:
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


def main():
    np.random.seed(42)

    model = SimpleModel()
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Training data: y = 2x + 3
    # Shape convention in your library:
    # x shape = (in_features, batch_size)
    x_data = np.array([[1.0, 2.0, 3.0, 4.0]])
    y_data = np.array([[5.0, 7.0, 9.0, 11.0]])

    x = Tensor(x_data, requires_grad=True)
    y = Tensor(y_data, requires_grad=False)

    epochs = 200

    print("Training started...\n")

    for epoch in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data}")

    print("\nTraining finished.")
    print("\nFinal loss:", loss.data)
    print("\nFinal predictions:")
    print(pred.data)

    print("\nTarget values:")
    print(y.data)

    print("\nLearned parameters:")
    print("L1 weight:\n", model.l1.weight.data)
    print("L1 bias:\n", model.l1.bias.data)
    print("L2 weight:\n", model.l2.weight.data)
    print("L2 bias:\n", model.l2.bias.data)


if __name__ == "__main__":
    main()