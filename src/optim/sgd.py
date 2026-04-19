class SGD:
    """
    Stochastic Gradient Descent optimizer.
    Updates parameters using: param -= lr * param.grad
    """

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Update all parameters.
        """
        for p in self.parameters:
            if p.grad is None:
                continue
            # Gradient descent update
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Reset gradients of all parameters.
        """
        for p in self.parameters:
            p.zero_grad()

