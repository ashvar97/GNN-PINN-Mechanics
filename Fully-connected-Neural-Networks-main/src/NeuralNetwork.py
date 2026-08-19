import copy
import SoftMax

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
        return self.loss_layer.forward(self.input_tensor, self.label_tensor)

    def backward(self):
        prev_layer = self.loss_layer.backward(self.label_tensor)

        for layer in self.layers[::-1]:
            prev_layer = layer.backward(prev_layer)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):
        for j in range(iterations):
            loss_fw = self.forward()
            self.loss.append(loss_fw)
            self.backward()

    def test(self, input_tensor):
        softmax = SoftMax.SoftMax()
        for i in self.layers:
            input_tensor = i.forward(input_tensor)
        return softmax.forward(input_tensor)
