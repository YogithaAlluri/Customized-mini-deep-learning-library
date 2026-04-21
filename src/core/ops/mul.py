from core.grad_function import GradFunction



class Mul(GradFunction):
    """
    Gradient function for multiplication: z = x * y
    """

    def backward(self, grad_output):
        x, y = self.saved_tensors

        # dz/dx = y
        grad_x = grad_output * y.data

        # dz/dy = x
        grad_y = grad_output * x.data

        return grad_x, grad_y


def mul(x, y):
    """
    Perform x * y with autograd support.
    """
    # Forward pass
    out_data = x.data * y.data

    # Create output tensor
    from core.tensor import Tensor
    out = Tensor(out_data, requires_grad=(x.requires_grad or y.requires_grad))

    # Attach grad function
    if out.requires_grad:
        grad_fn = Mul()
        grad_fn.save_for_backward(x, y)
        out.set_grad_fn(grad_fn, parents=[x, y])

    return out
