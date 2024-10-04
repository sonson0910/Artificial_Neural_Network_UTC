import pickle as pkl


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)
    
    def setup_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        n = len(input)
        result = []
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
        
    def fit(self, x_train, y_train, epochs, learning_rate):
        n = len(x_train)
        for i in range(epochs):
            err = 0 # total error
            for j in range(n):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)
                # compute derivative of the loss function
                error = self.loss_prime(y_train[j], output)
                # back propagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= n # average error
            print(f'epoch {i+1}/{epochs} error={err}')

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self = pkl.load(f)
        return self
            