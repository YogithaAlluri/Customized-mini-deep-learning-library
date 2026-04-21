import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        A minimal Tensor class that wraps a NumPy array and
        optionally tracks gradients.
        """
        # Store the actual numerical data as a NumPy array
        self.data = np.array(data, dtype=float)

        # Whether this tensor should track gradients
        self.requires_grad = requires_grad

        # This will hold the gradient of some scalar loss w.r.t this tensor
        self.grad = None

        # This will later point to a GradFunction that created this tensor
        self.grad_fn = None

        # Parents in the computation graph
        self.parents = []

    def zero_grad(self):
        """Reset the gradient to None."""
        self.grad = None

    def detach(self):
        """Return a new Tensor with same data but no gradient tracking."""
        return Tensor(self.data, requires_grad=False)

    def set_grad_fn(self, grad_fn, parents):
        """Attach grad_fn and parent tensors."""
        self.grad_fn = grad_fn
        self.parents = parents

    # -----------------------------
    # AUTOGRAD BACKWARD IMPLEMENTATION
    # -----------------------------
    def backward(self, grad_output=None):
        """
        Compute gradients for all tensors in the computation graph.
        """

        # If this is the final scalar, gradient = 1
        if grad_output is None:
            grad_output = 1.0

        # Initialize gradient for this tensor
        self.grad = grad_output

        # Stack for graph traversal
        stack = [self]

        while stack:
            t = stack.pop()

            # If no grad_fn, nothing to backpropagate
            if t.grad_fn is None:
                continue

            # Call backward of the operation
            grads = t.grad_fn.backward(t.grad)

            # grads is a tuple: (grad_x, grad_y, ...)
            for parent, grad in zip(t.parents, grads):

                # Accumulate gradient
                if parent.grad is None:
                    parent.grad = grad
                else:
                    parent.grad += grad

                # Continue backprop if parent has a grad_fn
                if parent.grad_fn is not None:
                    stack.append(parent)

    # -----------------------------
    # OPERATORS
    # -----------------------------
    def __add__(self, other):
        from core.ops.add import add
        return add(self, other)

    def __mul__(self, other):
        from core.ops.mul import mul
        return mul(self, other)

    def __matmul__(self, other):
        from core.ops.matmul import matmul
        return matmul(self, other)
    def __sub__(self, other):
        from core.ops.add import add
        from core.ops.mul import mul
        return add(self, mul(other, Tensor(-1.0)))


    
    @property
    def T(self):
       return Tensor(self.data.T, requires_grad=self.requires_grad)



 
