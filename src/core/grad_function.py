class GradFunction:
    """
    Base class for all gradient functions.
    Each operation (Add, Mul, MatMul, ReLU, etc.)
    will create a subclass of this.
    """

    def __init__(self):
        # Tensors saved for backward pass
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        """
        Store tensors needed to compute gradients later.
        """
        self.saved_tensors = tensors

    def backward(self, grad_output):
        """
        Compute gradients of the inputs given the gradient of the output.
        Subclasses will override this.
        """
        raise NotImplementedError("Backward not implemented")

