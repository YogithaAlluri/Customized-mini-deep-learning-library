import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from core.tensor import Tensor
from nn.linear import Linear
from nn.relu import ReLU
from nn.mse import MSELoss
from optim.sgd import SGD


# Simple model: Linear → ReLU → Linear
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


# Create model, loss, optimizer
model = SimpleModel()
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Dummy training data
x = Tensor(np.array([[1.0]]), requires_grad=True)
y = Tensor(np.array([[2.0]]), requires_grad=False)

print("l1.weight:", model.l1.weight.data.shape)
print("l1.bias:", model.l1.bias.data.shape)
print("l2.weight:", model.l2.weight.data.shape)
print("l2.bias:", model.l2.bias.data.shape)


# Training loop
for step in range(5):
    # Forward pass
    pred = model(x)
    loss = loss_fn(pred, y)

    print(f"Step {step}, Loss = {loss.data}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()
