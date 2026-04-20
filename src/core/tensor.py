import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, (int, float, list)):
            self.data = np.array(data, dtype=float)
        else:
            self.data = data.astype(float)

        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.parents = []

    def zero_grad(self):
        self.grad = None

    def set_grad_fn(self, grad_fn, parents):
        self.grad_fn = grad_fn
        self.parents = parents

    def backward(self, grad_output=None):
        if grad_output is None:
            # Loss must be a scalar for default backward
            grad_output = np.ones_like(self.data)
        
        self.grad = grad_output

        # 1. Build Topological Sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # 2. Walk backwards through the graph
        for v in reversed(topo):
            if v.grad_fn is None:
                continue

            grads = v.grad_fn.backward(v.grad)
            
            # Standardize to tuple
            if not isinstance(grads, (tuple, list)):
                grads = (grads,)

            for parent, grad in zip(v.parents, grads):
                if parent.requires_grad:
                    if parent.grad is None:
                        # Copy ensures no accidental in-place corruption
                        parent.grad = np.array(grad, copy=True)
                    else:
                        parent.grad += grad

    def __add__(self, other):
        from core.ops.add import add
        other = other if isinstance(other, Tensor) else Tensor(other)
        return add(self, other)

    def __mul__(self, other):
        from core.ops.mul import mul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return mul(self, other)

    def __matmul__(self, other):
        from core.ops.matmul import matmul
        return matmul(self, other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * -1.0)

    @property
    def T(self):
        # Note: Ideally this should be a Transpose Op, 
        # but this allows basic shape manipulation
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"