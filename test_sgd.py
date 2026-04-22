import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from core.parameter import Parameter
from optim.sgd import SGD


def test_sgd_parameter_update():
    # Create a parameter manually
    p = Parameter(np.array([[2.0, 4.0]]))

    # Assign fake gradient
    p.grad = np.array([[1.0, 1.0]])

    optimizer = SGD([p], lr=0.1)

    old_value = p.data.copy()

    optimizer.step()

    expected = old_value - 0.1 * np.array([[1.0, 1.0]])

    print("Old value:\n", old_value)
    print("New value:\n", p.data)
    print("Expected:\n", expected)

    assert np.allclose(p.data, expected)

    print("test_sgd_parameter_update passed")


def test_sgd_no_grad():
    p = Parameter(np.array([[2.0, 4.0]]))

    # No gradient assigned → should not change
    p.grad = None

    optimizer = SGD([p], lr=0.1)

    old_value = p.data.copy()

    optimizer.step()

    print("Old value:\n", old_value)
    print("New value:\n", p.data)

    assert np.allclose(p.data, old_value)

    print("test_sgd_no_grad passed")


def test_sgd_zero_grad():
    p = Parameter(np.array([[2.0, 4.0]]))

    p.grad = np.array([[1.0, 2.0]])

    optimizer = SGD([p], lr=0.1)

    optimizer.zero_grad()

    print("Gradient after zero_grad:", p.grad)

    assert p.grad is None

    print("test_sgd_zero_grad passed")


def test_sgd_multiple_parameters():
    p1 = Parameter(np.array([[1.0, 2.0]]))
    p2 = Parameter(np.array([[3.0, 4.0]]))

    p1.grad = np.array([[1.0, 1.0]])
    p2.grad = np.array([[2.0, 2.0]])

    optimizer = SGD([p1, p2], lr=0.1)

    old_p1 = p1.data.copy()
    old_p2 = p2.data.copy()

    optimizer.step()

    expected_p1 = old_p1 - 0.1 * p1.grad
    expected_p2 = old_p2 - 0.1 * p2.grad

    print("p1 new:", p1.data, "expected:", expected_p1)
    print("p2 new:", p2.data, "expected:", expected_p2)

    assert np.allclose(p1.data, expected_p1)
    assert np.allclose(p2.data, expected_p2)

    print("test_sgd_multiple_parameters passed")


if __name__ == "__main__":
    test_sgd_parameter_update()
    test_sgd_no_grad()
    test_sgd_zero_grad()
    test_sgd_multiple_parameters()

    print("\nAll SGD tests passed successfully.")