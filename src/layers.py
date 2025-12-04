import numpy as np

class Identity:
    def forward(self, x):
        self.x = x
        return x
    
    def backward(self, grad_output):
        return grad_output

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.x > 0)

class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class Sigmoid:
    def forward(self, x):
        # Clip to prevent overflow 
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Linear:
    def __init__(self, input_size, output_size, activation=None):
        # He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation if activation else Identity()
        
        # Gradient accumulators
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        z = x @ self.weights + self.bias
        self.output = self.activation.forward(z)
        return self.output
    
    def backward(self, grad_output):
        grad = self.activation.backward(grad_output)
        
        self.grad_weights += self.input.T @ grad
        self.grad_bias += np.sum(grad, axis=0, keepdims=True)
        
        return grad @ self.weights.T
    
    def zero_grad(self):
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)
    
    def update(self, learning_rate):
        # Gradient clipping (Norm > 10.0)
        grad_norm = np.sqrt(np.sum(self.grad_weights**2) + np.sum(self.grad_bias**2))
        if grad_norm > 10.0: 
            self.grad_weights = self.grad_weights * (10.0 / grad_norm)
            self.grad_bias = self.grad_bias * (10.0 / grad_norm)
        
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias