class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def backward(self):
        pass

    def zero_grad(self):
        pass

    def detach(self):
        pass

