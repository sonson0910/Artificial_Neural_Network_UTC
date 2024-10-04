from .layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        """_summary_

        Args:
            input_shape (_type_): (1, 3)
            output_shape (_type_): (1, 4)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5 # (input_shape[1], output_shape[1]) = (3, 4)
        self.bias = np.random.rand(1, output_shape[1]) - 0.5 # (1, 4)

    def forward_propagation(self, input):
        self.input = input.reshape(1, -1)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        current_layer_error = np.dot(output_error, self.weights.T) # (1, 4) | x error
        weights_error = np.dot(self.input.T, output_error) # (1, 3) | w error
        bias_error = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return current_layer_error