import torch
class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size) * 0.01
        self.bias = torch.zeros(output_size)
        self.input = None

    def forward(self, x):
        self.input = x
        return torch.matmul(x, self.weights) + self.bias

    def backward(self, grad_output):
        grad_input = grad_output.matmul(self.weights.T)
        grad_weights = self.input.T.matmul(grad_output)
        grad_bias = grad_output.sum(dim=0)

        self.weights.grad = grad_weights
        self.bias.grad = grad_bias

        return grad_input


class SigmoidLayer:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return 1 / (1 + torch.exp(-x))

    def backward(self, grad_output):
        sigmoid = self.forward(self.input)
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        return grad_input


class ReLULayer:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return torch.maximum(x, torch.tensor(0.0))

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input[self.input < 0] = 0
        return grad_input


class DropoutLayer:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, x):
        self.mask = (torch.rand_like(x) > self.p).float()
        return x * self.mask / (1 - self.p)

    def backward(self, grad_output):
        return grad_output * self.mask / (1 - self.p)
