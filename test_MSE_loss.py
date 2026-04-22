import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from core.tensor import Tensor
from nn.mse import MSELoss


def test_mse_forward_value():
    loss_fn = MSELoss()

    y_pred = Tensor(np.array([
        [2.0],
        [4.0]
    ]), requires_grad=True)

    y_true = Tensor(np.array([
        [1.0],
        [1.0]
    ]), requires_grad=False)

    loss = loss_fn(y_pred, y_true)

    # diff = [[1], [3]]
    # sq   = [[1], [9]]
    # mean = 5
    expected = 5.0

    print("Computed loss:", loss.data)
    print("Expected loss:", expected)

    assert np.isclose(loss.data, expected)
    print("test_mse_forward_value passed")


def test_mse_output_is_scalar():
    loss_fn = MSELoss()

    y_pred = Tensor(np.array([
        [1.0],
        [2.0],
        [3.0]
    ]), requires_grad=True)

    y_true = Tensor(np.array([
        [1.0],
        [2.0],
        [2.0]
    ]), requires_grad=False)

    loss = loss_fn(y_pred, y_true)

    assert np.isscalar(loss.data) or np.array(loss.data).shape == ()
    print("Loss output:", loss.data)
    print("test_mse_output_is_scalar passed")


def test_mse_backward_gradient_y_pred():
    loss_fn = MSELoss()

    y_pred = Tensor(np.array([
        [2.0],
        [4.0]
    ]), requires_grad=True)

    y_true = Tensor(np.array([
        [1.0],
        [1.0]
    ]), requires_grad=False)

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    # n = 2
    # dL/dy_pred = 2*(y_pred - y_true)/n
    #            = 2*[[1],[3]] / 2
    #            = [[1],[3]]
    expected_grad = np.array([
        [1.0],
        [3.0]
    ])

    print("Computed y_pred.grad:\n", y_pred.grad)
    print("Expected y_pred.grad:\n", expected_grad)

    assert y_pred.grad is not None
    assert np.allclose(y_pred.grad, expected_grad)
    print("test_mse_backward_gradient_y_pred passed")


def test_mse_backward_gradient_y_true():
    loss_fn = MSELoss()

    y_pred = Tensor(np.array([
        [2.0],
        [4.0]
    ]), requires_grad=True)

    y_true = Tensor(np.array([
        [1.0],
        [1.0]
    ]), requires_grad=False)

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    # dL/dy_true = -dL/dy_pred
    expected_grad = np.array([
        [-1.0],
        [-3.0]
    ])

    print("Computed y_true.grad:\n", y_true.grad)
    print("Expected y_true.grad:\n", expected_grad)

    assert y_true.grad is not None
    assert np.allclose(y_true.grad, expected_grad)
    print("test_mse_backward_gradient_y_true passed")


def test_mse_gradient_shapes():
    loss_fn = MSELoss()

    y_pred = Tensor(np.array([
        [2.0],
        [4.0],
        [6.0]
    ]), requires_grad=True)

    y_true = Tensor(np.array([
        [1.0],
        [1.0],
        [1.0]
    ]), requires_grad=False)

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    assert y_pred.grad.shape == y_pred.data.shape
    assert y_true.grad.shape == y_true.data.shape

    print("y_pred.grad shape:", y_pred.grad.shape)
    print("y_true.grad shape:", y_true.grad.shape)
    print("test_mse_gradient_shapes passed")


def test_mse_zero_loss_case():
    loss_fn = MSELoss()

    y_pred = Tensor(np.array([
        [1.0],
        [2.0],
        [3.0]
    ]), requires_grad=True)

    y_true = Tensor(np.array([
        [1.0],
        [2.0],
        [3.0]
    ]), requires_grad=False)

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    expected_loss = 0.0
    expected_grad = np.array([
        [0.0],
        [0.0],
        [0.0]
    ])

    print("Computed zero-loss case:", loss.data)
    print("Computed y_pred.grad:\n", y_pred.grad)

    assert np.isclose(loss.data, expected_loss)
    assert np.allclose(y_pred.grad, expected_grad)
    assert np.allclose(y_true.grad, expected_grad)

    print("test_mse_zero_loss_case passed")


if __name__ == "__main__":
    test_mse_forward_value()
    test_mse_output_is_scalar()
    test_mse_backward_gradient_y_pred()
    test_mse_backward_gradient_y_true()
    test_mse_gradient_shapes()
    test_mse_zero_loss_case()

    print("\nAll MSELoss tests passed successfully.")