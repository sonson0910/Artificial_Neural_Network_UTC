import numpy as np
from .layer import Layer

class DropoutLayer(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward_propagation(self, input):
        self.mask = np.random.binomial(1, self.dropout_rate, size=input.shape)
        self.output = input * self.mask
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.mask
