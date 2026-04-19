from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum

    def step(self):
        pass
