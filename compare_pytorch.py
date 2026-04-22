import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from core.tensor import Tensor
from nn.linear import Linear
from nn.relu import ReLU
from nn.mse import MSELoss
from optim.sgd import SGD


# -----------------------------
# YOUR LIBRARY MODEL
# -----------------------------
class MyModel:
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


# -----------------------------
# PYTORCH MODEL
# -----------------------------
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Dataset
    x_np = np.array([[1.0, 2.0, 3.0, 4.0]])
    y_np = np.array([[5.0, 7.0, 9.0, 11.0]])

    # -----------------------------
    # YOUR LIBRARY
    # -----------------------------
    my_model = MyModel()
    my_loss_fn = MSELoss()
    my_optimizer = SGD(my_model.parameters(), lr=0.01)

    x_my = Tensor(x_np, requires_grad=True)
    y_my = Tensor(y_np, requires_grad=False)

    my_losses = []

    for epoch in range(200):
        pred = my_model(x_my)
        loss = my_loss_fn(pred, y_my)

        my_optimizer.zero_grad()
        loss.backward()
        my_optimizer.step()

        my_losses.append(loss.data)

    # -----------------------------
    # PYTORCH
    # -----------------------------
    torch_model = TorchModel()
    torch_loss_fn = nn.MSELoss()
    torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.01)

    # reshape to (batch_size, features)
    x_torch = torch.tensor(x_np.T, dtype=torch.float32)
    y_torch = torch.tensor(y_np.T, dtype=torch.float32)

    torch_losses = []

    for epoch in range(200):
        pred = torch_model(x_torch)
        loss = torch_loss_fn(pred, y_torch)

        torch_optimizer.zero_grad()
        loss.backward()
        torch_optimizer.step()

        torch_losses.append(loss.item())

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure()
    plt.plot(my_losses, label="My Library")
    plt.plot(torch_losses, label="PyTorch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison: Custom vs PyTorch")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()