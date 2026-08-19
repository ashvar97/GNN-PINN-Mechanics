import numpy as np
import Base
class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.trainable = True
        self.weights = np.random.uniform(0 ,1 ,size = (input_size, output_size))
        self.bias = np.random.uniform(0, 1, size=(1 ,output_size))

    def optimizer(self):
        return self._optimizer

    def optimizer(self, current_opt):
        self._optimizer = current_opt

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        next_tensor = (self.input_tensor @ self.weights) + self.bias
        return next_tensor

    def backward(self, error_tensor):
        error_tensor_prev = error_tensor @ np.transpose(self.weights)
        self.gradient_weights = np.transpose(self.input_tensor) @ error_tensor
        try:
            upd = self.optimizer
            self.weights = upd.calculate_update(self.weights, self.gradient_weights)
            self.bias = upd.calculate_update(self.bias,  error_tensor)
        except AttributeError:
            pass
        return error_tensor_prev

