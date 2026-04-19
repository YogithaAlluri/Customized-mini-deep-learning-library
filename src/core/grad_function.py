class GradFunction:
    def __init__(self, parents):
        self.parents = parents
        self.saved_tensors = ()

    def backward(self, grad_output):
        pass

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
