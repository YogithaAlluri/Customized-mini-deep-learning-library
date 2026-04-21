from core.tensor import Tensor


class Parameter(Tensor):
    """
    A trainable parameter in a neural network.
    Behaves like a Tensor but always requires gradients.
    """

    def __init__(self, data):
        # Always requires_grad=True for parameters
        super().__init__(data, requires_grad=True)

