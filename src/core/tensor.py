import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        # Ensure data is a float numpy array
        if isinstance(data, (int, float, list)):
            self.data = np.array(data, dtype=float)
        else:
            self.data = data.astype(float)

        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
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
        Compute gradients using Topological Sort to ensure correct accumulation.
        """
        # If this is the starting scalar (Loss), gradient is 1.0
        if grad_output is None:
            if self.data.size > 1:
                raise RuntimeError("Backward can only be called on a scalar (size 1) without grad_output.")
            grad_output = np.ones_like(self.data)

        self.grad = grad_output

        # 1. Build the topological order of the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Traverse parents first
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # 2. Process nodes in reverse topological order (backwards)
        for v in reversed(topo):
            if v.grad_fn is None:
                continue

            # Call the backward operation of the specific GradFunction
            # This should return a tuple of gradients (one for each parent)
            grads = v.grad_fn.backward(v.grad)

            # Standardize grads into a tuple if it's a single array
            if not isinstance(grads, (tuple, list)):
                grads = (grads,)

            # Accumulate gradients into parents
            for parent, grad in zip(v.parents, grads):
                if parent.requires_grad:
                    if parent.grad is None:
                        # Use np.copy to prevent accidental in-place mutations across paths
                        parent.grad = np.array(grad, copy=True)
                    else:
                        # Accumulate: dL/dx = sum(dL/dy * dy/dx)
                        parent.grad += grad

    # -----------------------------
    # OPERATORS (Updated with better import handling)
    # -----------------------------
    def __add__(self, other):
        from core.ops.add import add
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return add(self, other)

    def __mul__(self, other):
        from core.ops.mul import mul
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return mul(self, other)

    def __matmul__(self, other):
        from core.ops.matmul import matmul
        return matmul(self, other)

    def __sub__(self, other):
        # x - y is equivalent to x + (-1 * y)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (other * -1.0)

    @property
    def T(self):
        # NOTE: For a perfect autograd, this should be a GradFunction (Transpose)
        # However, this returns a new Tensor for basic usage.
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"