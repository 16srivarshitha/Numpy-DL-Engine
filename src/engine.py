import numpy as np

class Model:
    def __init__(self, layers, loss_type='mse', learning_rate=0.001):
        self.layers = layers
        self.loss_type = loss_type
        self.learning_rate = learning_rate
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def compute_loss(self, y_pred, y_true):
        if self.loss_type == 'mse':
            loss = np.mean((y_pred - y_true) ** 2)
            grad_loss = 2 * (y_pred - y_true) / y_pred.shape[0]
            
        elif self.loss_type == 'bce':
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            grad_loss = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_pred.shape[0]
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss, grad_loss
    
    def train_step(self, x, y):
        y_pred = self.forward(x)
        loss, grad_loss = self.compute_loss(y_pred, y)
        self.backward(grad_loss)
        return loss
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate)
    
    def predict(self, x):
        output = self.forward(x)
        return np.clip(output, 0, 1)
    
    def save_to(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'layer_{i}_weights'] = layer.weights
            params[f'layer_{i}_bias'] = layer.bias
        np.savez(path, **params)
    
    def count_parameters(self):
        total = 0
        for layer in self.layers:
            total += layer.weights.size + layer.bias.size
        return total