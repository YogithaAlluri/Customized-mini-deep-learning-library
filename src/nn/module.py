from core.parameter import Parameter


class Module:
    """
    Base class for all neural network modules (layers, models, etc.)
    """

    def parameters(self):
        """
        Return all Parameter objects inside this module.
        This includes parameters of submodules.
        """
        params = []

        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            # If it's a Parameter, collect it
            if isinstance(attr, Parameter):
                params.append(attr)

            # If it's a Module, collect its parameters too
            if isinstance(attr, Module):
                params.extend(attr.parameters())

        return params

    def zero_grad(self):
        """
        Reset gradients of all parameters.
        """
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        """
        Allows the module to be called like a function.
        Example: output = layer(input)
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Forward method not implemented")


