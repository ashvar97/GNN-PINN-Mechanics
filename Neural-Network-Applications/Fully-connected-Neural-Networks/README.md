# Fully Connected Neural Networks (FCNN) with PyTorch



A PyTorch implementation of Fully Connected Neural Networks (FCNN) for classification tasks, featuring modular components for flexible architecture design.

## 📂 Repository Structure

src/
├── Base.py              # Base classes for network components (layer templates, core functionality)  
├── FullyConnected.py    # Implements fully connected/dense layers (input/output size configurable)  
├── Loss.py             # Loss functions: CrossEntropy (classification), MSE (regression)  
├── NeuralNetwork.py    # Main class to assemble layers into a trainable network  
├── Optimizers.py       # Gradient descent optimizers: SGD (with momentum), Adam  
├── ReLU.py            # Rectified Linear Unit (ReLU) activation layer implementation  
└── SoftMax.py         # Softmax activation for multi-class classification outputs  





## ✨ Features

- **Modular Design**: Each component (layers, activations, losses) is independently usable
- **PyTorch Integration**: Leverages PyTorch's automatic differentiation
- **Flexible Architectures**: Easily configurable hidden layers and sizes
- **Multiple Activations**: ReLU and Softmax included
- **Extensible**: Add new layers or losses with base class templates

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ashvar97/Fully-connected-Neural-Networks.git
   cd Fully-connected-Neural-Networks




