import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from core.tensor import Tensor
from nn.linear import Linear
from nn.relu import ReLU
from nn.mse import MSELoss
from optim.sgd import SGD


def test_linear_initialization():
    layer = Linear(3, 2)

    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")

    assert layer.weight.requires_grad is True
    assert layer.bias.requires_grad is True

    assert layer.weight.data.shape == (2, 3)
    assert layer.bias.data.shape == (2, 1)

    params = layer.parameters()
    assert len(params) == 2

    print("test_linear_initialization passed")


def test_linear_forward_shape():
    layer = Linear(3, 2)

    x = Tensor(np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]), requires_grad=True)

    out = layer(x)

    assert out.data.shape == (2, 4)

    print("test_linear_forward_shape passed")


def test_linear_forward_values():
    layer = Linear(2, 2)

    layer.weight.data = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    layer.bias.data = np.array([
        [0.5],
        [-0.5]
    ])

    x = Tensor(np.array([
        [1.0],
        [2.0]
    ]), requires_grad=True)

    out = layer(x)

    expected = np.array([
        [5.5],
        [10.5]
    ])

    assert np.allclose(out.data, expected)

    print("Computed output:\n", out.data)
    print("Expected output:\n", expected)
    print("test_linear_forward_values passed")


def test_linear_backward_gradients_exist():
    layer = Linear(2, 1)
    loss_fn = MSELoss()

    x = Tensor(np.array([
        [1.0],
        [2.0]
    ]), requires_grad=True)

    target = Tensor(np.array([
        [3.0]
    ]), requires_grad=False)

    pred = layer(x)
    loss = loss_fn(pred, target)
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert pred.grad is not None

    assert layer.weight.grad.shape == layer.weight.data.shape
    assert layer.bias.grad.shape == layer.bias.data.shape

    print("weight.grad:\n", layer.weight.grad)
    print("bias.grad:\n", layer.bias.grad)
    print("test_linear_backward_gradients_exist passed")


def test_linear_optimizer_update():
    layer = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(layer.parameters(), lr=0.01)

    layer.weight.data = np.array([[1.0, 2.0]])
    layer.bias.data = np.array([[0.5]])

    x = Tensor(np.array([
        [1.0],
        [2.0]
    ]), requires_grad=True)

    target = Tensor(np.array([
        [1.0]
    ]), requires_grad=False)

    old_weight = layer.weight.data.copy()
    old_bias = layer.bias.data.copy()

    pred = layer(x)
    loss = loss_fn(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("old weight:\n", old_weight)
    print("new weight:\n", layer.weight.data)
    print("old bias:\n", old_bias)
    print("new bias:\n", layer.bias.data)

    assert not np.allclose(layer.weight.data, old_weight)
    assert not np.allclose(layer.bias.data, old_bias)

    print("test_linear_optimizer_update passed")


def test_relu_forward_values():
    relu = ReLU()

    x = Tensor(np.array([
        [-2.0],
        [0.0],
        [3.5],
        [-1.2],
        [4.0]
    ]), requires_grad=True)

    out = relu(x)

    expected = np.array([
        [0.0],
        [0.0],
        [3.5],
        [0.0],
        [4.0]
    ])

    assert np.allclose(out.data, expected)

    print("Computed ReLU output:\n", out.data)
    print("Expected ReLU output:\n", expected)
    print("test_relu_forward_values passed")


def test_relu_shape_preserved():
    relu = ReLU()

    x = Tensor(np.array([
        [1.0, -2.0, 3.0],
        [-4.0, 5.0, -6.0]
    ]), requires_grad=True)

    out = relu(x)

    assert out.data.shape == x.data.shape

    print("test_relu_shape_preserved passed")


def test_relu_backward_mask():
    relu = ReLU()
    loss_fn = MSELoss()

    x = Tensor(np.array([
        [-1.0],
        [2.0],
        [-3.0],
        [4.0]
    ]), requires_grad=True)

    out = relu(x)

    target = Tensor(np.array([
        [0.0],
        [0.0],
        [0.0],
        [0.0]
    ]), requires_grad=False)

    loss = loss_fn(out, target)
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape

    assert np.isclose(x.grad[0, 0], 0.0)
    assert np.isclose(x.grad[2, 0], 0.0)

    print("x.grad:\n", x.grad)
    print("test_relu_backward_mask passed")


if __name__ == "__main__":
    test_linear_initialization()
    test_linear_forward_shape()
    test_linear_forward_values()
    test_linear_backward_gradients_exist()
    test_linear_optimizer_update()
    test_relu_forward_values()
    test_relu_shape_preserved()
    test_relu_backward_mask()

    print("\nAll layer tests passed successfully.")