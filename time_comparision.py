import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import time
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
# YOUR MODEL
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
        return self.l2(self.relu(self.l1(x)))


# -----------------------------
# MAIN
# -----------------------------
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Dataset
    x_np = np.array([[1.0, 2.0, 3.0, 4.0]])
    y_np = np.array([[5.0, 7.0, 9.0, 11.0]])

    # -----------------------------
    # MY LIBRARY TIMING
    # -----------------------------
    my_model = MyModel()
    my_loss = MSELoss()
    my_opt = SGD(my_model.parameters(), lr=0.01)

    x_my = Tensor(x_np, requires_grad=True)
    y_my = Tensor(y_np, requires_grad=False)

    start = time.time()

    for epoch in range(200):
        pred = my_model(x_my)
        loss = my_loss(pred, y_my)

        my_opt.zero_grad()
        loss.backward()
        my_opt.step()

    my_time = time.time() - start

    # -----------------------------
    # PYTORCH TIMING
    # -----------------------------
    torch_model = TorchModel()
    torch_loss = nn.MSELoss()
    torch_opt = optim.SGD(torch_model.parameters(), lr=0.01)

    x_torch = torch.tensor(x_np.T, dtype=torch.float32)
    y_torch = torch.tensor(y_np.T, dtype=torch.float32)

    start = time.time()

    for epoch in range(200):
        pred = torch_model(x_torch)
        loss = torch_loss(pred, y_torch)

        torch_opt.zero_grad()
        loss.backward()
        torch_opt.step()

    torch_time = time.time() - start

    # -----------------------------
    # PRINT RESULTS
    # -----------------------------
    print("My Library Time:", my_time)
    print("PyTorch Time:", torch_time)

    # -----------------------------
    # PLOT BAR CHART
    # -----------------------------
    plt.figure()
    plt.bar(["My Library", "PyTorch"], [my_time, torch_time])
    plt.ylabel("Time (seconds)")
    plt.title("Training Time Comparison")
    plt.show()


if __name__ == "__main__":
    main()