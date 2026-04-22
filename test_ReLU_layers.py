import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from core.tensor import Tensor
from nn.relu import ReLU
from nn.mse import MSELoss


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


def test_relu_all_negative():
    relu = ReLU()

    x = Tensor(np.array([
        [-1.0],
        [-2.0],
        [-3.0]
    ]), requires_grad=True)

    out = relu(x)

    expected = np.array([
        [0.0],
        [0.0],
        [0.0]
    ])

    assert np.allclose(out.data, expected)

    print("Computed ReLU output (all negative):\n", out.data)
    print("Expected output:\n", expected)
    print("test_relu_all_negative passed")


def test_relu_all_positive():
    relu = ReLU()

    x = Tensor(np.array([
        [1.0],
        [2.5],
        [3.0]
    ]), requires_grad=True)

    out = relu(x)

    expected = np.array([
        [1.0],
        [2.5],
        [3.0]
    ])

    assert np.allclose(out.data, expected)

    print("Computed ReLU output (all positive):\n", out.data)
    print("Expected output:\n", expected)
    print("test_relu_all_positive passed")


def test_relu_zero_input():
    relu = ReLU()

    x = Tensor(np.array([
        [0.0],
        [0.0],
        [0.0]
    ]), requires_grad=True)

    out = relu(x)

    expected = np.array([
        [0.0],
        [0.0],
        [0.0]
    ])

    assert np.allclose(out.data, expected)

    print("Computed ReLU output (all zeros):\n", out.data)
    print("Expected output:\n", expected)
    print("test_relu_zero_input passed")


def test_relu_backward_all_negative():
    relu = ReLU()
    loss_fn = MSELoss()

    x = Tensor(np.array([
        [-1.0],
        [-2.0],
        [-3.0]
    ]), requires_grad=True)

    out = relu(x)

    target = Tensor(np.array([
        [0.0],
        [0.0],
        [0.0]
    ]), requires_grad=False)

    loss = loss_fn(out, target)
    loss.backward()

    expected_grad = np.array([
        [0.0],
        [0.0],
        [0.0]
    ])

    assert x.grad is not None
    assert np.allclose(x.grad, expected_grad)

    print("x.grad (all negative input):\n", x.grad)
    print("test_relu_backward_all_negative passed")


def test_relu_backward_all_positive():
    relu = ReLU()
    loss_fn = MSELoss()

    x = Tensor(np.array([
        [1.0],
        [2.0]
    ]), requires_grad=True)

    out = relu(x)

    target = Tensor(np.array([
        [0.0],
        [0.0]
    ]), requires_grad=False)

    loss = loss_fn(out, target)
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape
    assert np.all(x.grad > 0)

    print("x.grad (all positive input):\n", x.grad)
    print("test_relu_backward_all_positive passed")


if __name__ == "__main__":
    test_relu_forward_values()
    test_relu_shape_preserved()
    test_relu_backward_mask()
    test_relu_all_negative()
    test_relu_all_positive()
    test_relu_zero_input()
    test_relu_backward_all_negative()
    test_relu_backward_all_positive()

    print("\nAll ReLU tests passed successfully.")