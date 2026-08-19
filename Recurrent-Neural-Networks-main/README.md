# Recurrent Neural Networks (RNN)

## Overview
This project implements a **Recurrent Neural Network (RNN)** for sequence-based tasks using PyTorch. It includes scripts for model training and evaluation.


# Modular Neural Network Framework


A lightweight, modular implementation of neural network components from scratch, including **RNNs, LSTMs, and traditional layers** (convolutional, pooling, dropout, etc.). Designed for educational purposes and custom deep learning projects.

## Project Structure
pip install numpy for numpy functions

├── Base.py                  # Base classes for neural network layers and core functionality  
├── BatchNormalization.py    # Implements batch normalization for stabilizing training  
├── Constraints.py           # Weight constraints (e.g., max norm, unit norm)  
├── Conv.py                  # Convolutional layer (1D/2D) for spatial feature extraction  
├── Dropout.py               # Dropout layer for regularization  
├── Flatten.py               # Flattens input tensors for dense layers  
├── FullyConnected.py        # Standard dense/fully connected layer  
├── Initializers.py          # Weight initializers (Glorot, He, random, zeros)  
├── LSTM.py                  # Long Short-Term Memory (LSTM) recurrent layer  
├── Loss.py                  # Loss functions (MSE, CrossEntropy, etc.)  
├── NeuralNetwork.py         # Main class to compose/build neural networks  
├── Optimizers.py            # Gradient-based optimizers (SGD, Adam, RMSprop)  
├── Pooling.py               # Pooling layers (MaxPooling, AveragePooling)  
├── RNN.py                   # Vanilla Recurrent Neural Network (RNN) layer  
├── ReLU.py                  # Rectified Linear Unit (ReLU) activation  
├── Sigmoid.py               # Sigmoid activation for binary outputs  
├── SoftMax.py               # Softmax activation for multi-class classification  
├── TanH.py                  # Hyperbolic Tangent (TanH) activation  


## Key Features
- **Modular Design**: Each component (layers, activations, optimizers) is self-contained.
- **From-Scratch Implementation**: No reliance on TensorFlow/PyTorch (pure NumPy recommended).
- **Supported Layers**:
  - Recurrent: `RNN`, `LSTM`
  - Traditional: `FullyConnected`, `Conv`, `Pooling`, `Dropout`, `BatchNorm`
- **Extensible**: Easily add new layers or optimizers by following the base class template.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ashvar97/Recurrent-Neural-Networks.git
   cd Recurrent-Neural-Networks
