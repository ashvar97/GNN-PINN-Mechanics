# Convolutional Neural Network (CNN)

A modular implementation of Convolutional Neural Networks from scratch, supporting custom architectures for image classification tasks.


## ✨ Key Features

- **Core CNN Components**:
  - 2D Convolutional layers (`Conv.py`)
  - Pooling layers (`Pooling.py`)
  - Activation functions (`ReLU.py`, `SoftMax.py`)
  
- **Training Infrastructure**:
  - Multiple optimizers (`Optimizers.py`)
  - Custom loss functions (`Loss.py`)
  - Weight initializers (`Initializers.py`)

- **Flexible Architecture**:
  - Stackable layers via `NeuralNetwork.py`
  - Configurable hyperparameters

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/ashvar97/Convolution-Neural-Network.git
   cd Convolution-Neural-Network

## Repository Structure
## 📂 Repository Structure

├── Conv.py              # 2D Convolutional layer implementation (supports padding/striding)  
├── Flatten.py           # Flatten layer for CNN-to-FC transition (preserves batch dimension)  
├── FullyConnected.py    # Dense/fully connected layer (configurable input/output sizes)  
├── Initializers.py      # Weight initialization schemes (He, Xavier, normal, uniform)  
├── Loss.py             # Loss functions: CrossEntropy (classification), MSE (regression)  
├── NeuralNetwork.py    # Main network class (manages layer stacking/forward/backward pass)  
├── Optimizers.py       # Gradient optimizers: SGD (with momentum), Adam, RMSprop  
├── Pooling.py          # Pooling layers: MaxPooling (most common), AveragePooling  
├── ReLU.py            # ReLU activation implementation (in-place operations supported)  
└── SoftMax.py         # Softmax activation for multi-class classification outputs  

**Author**: Ashwin Varkey  
**Contact**: [ashvar97@gmail.com](mailto:ashvar97@gmail.com) | [LinkedIn](https://www.linkedin.com/in/ashvar97/)
