class Sgd:
    def __init__(self, learning):
        self.learning_rate = learning

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_wts = weight_tensor - self.learning_rate * gradient_tensor
        return updated_wts
