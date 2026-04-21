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


model = SimpleModel()
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

x = Tensor(np.array([[1.0]]), requires_grad=True)
y = Tensor(np.array([[2.0]]), requires_grad=False)

print("l1.weight shape:", model.l1.weight.data.shape)
print("l1.bias shape:", model.l1.bias.data.shape)
print("l2.weight shape:", model.l2.weight.data.shape)
print("l2.bias shape:", model.l2.bias.data.shape)
print("model.l1.weight id =", id(model.l1.weight))
print("model.parameters() =", model.parameters())
print("len(model.parameters()) =", len(model.parameters()))
print("l1.parameters() =", model.l1.parameters())
print("len(l1.parameters()) =", len(model.l1.parameters()))
print("l2.parameters() =", model.l2.parameters())
print("len(l2.parameters()) =", len(model.l2.parameters()))
print("type of model.l1.weight.data =", type(model.l1.weight.data))
for step in range(5):
    pred = model(x)
    loss = loss_fn(pred, y)

    print(f"\nStep {step}, Loss = {loss.data}")
    print("Before update l1.weight =", model.l1.weight.data)

    optimizer.zero_grad()
    loss.backward()
    print("type of l1.weight.grad =", type(model.l1.weight.grad))
    print("l1.weight.grad =", model.l1.weight.grad)
    print("l2.weight.grad =", model.l2.weight.grad)

    optimizer.step()

    print("After update l1.weight =", model.l1.weight.data)
