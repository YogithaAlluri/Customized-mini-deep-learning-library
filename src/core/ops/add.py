from core.grad_function import GradFunction
from core.tensor import Tensor


class Add(GradFunction):
    """
    Gradient function for addition: z = x + y
    """

    def backward(self, grad_output):
        # For addition, gradient flows unchanged to both inputs
        return grad_output, grad_output


def add(x, y):
    """
    Perform x + y with autograd support.
    """
    # Forward pass
    out_data = x.data + y.data

    # Create output tensor
    out = Tensor(out_data, requires_grad=(x.requires_grad or y.requires_grad))

    # Attach grad function
    if out.requires_grad:
        grad_fn = Add()
        grad_fn.save_for_backward(x, y)
        out.set_grad_fn(grad_fn, parents=[x, y])

    return out

