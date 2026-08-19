# Recurrent Neural Networks

A from-scratch NumPy deep learning framework (no PyTorch/TensorFlow) covering
fully-connected, convolutional, and recurrent architectures, plus a small
working application (`demo_delayed_echo.py`) that actually trains the RNN
on a real task rather than leaving it as unexercised library code.

It covers the same layer/optimizer building blocks as the sibling
[`../Convolution-Neural-Network`](../Convolution-Neural-Network) and
[`../Fully-connected-Neural-Networks`](../Fully-connected-Neural-Networks)
folders, plus the RNN/LSTM layers, as one complete, runnable package.

## Structure

The code is organized into real packages matching its internal imports
(`from Layers import Base`, `from Optimization import Optimizers`, ...):

```
Recurrent-Neural-Networks/
├── NeuralNetwork.py             # composes layers into a trainable network
├── demo_delayed_echo.py         # small application: RNN learns a memory task (see below)
├── Layers/
│   ├── Base.py                    # BaseLayer: shared layer interface
│   ├── FullyConnected.py
│   ├── Conv.py                     # 2D convolution (im2col via scipy.signal)
│   ├── Pooling.py
│   ├── Flatten.py
│   ├── ReLU.py / SoftMax.py / Sigmoid.py / TanH.py
│   ├── Dropout.py
│   ├── BatchNormalization.py
│   ├── RNN.py / LSTM.py
│   └── Helpers.py                 # see note below
└── Optimization/
    ├── Optimizers.py              # SGD, SGD+Momentum, Adam (+ regularizer hook)
    ├── Loss.py                     # CrossEntropy
    ├── Constraints.py              # L1/L2 weight regularizers
    └── Initializers.py            # Constant, UniformRandom, Xavier, He
```

`Layers/Helpers.py` provides `compute_bn_gradients`, used by
`BatchNormalization.py`, implemented using the standard analytic batch-norm
backward-pass formula (Ioffe & Szegedy, 2015); see the docstring in that file.

## RNN training details

`Layers/RNN.py` implements truncated backpropagation-through-time, with
gradients accumulated (`+=`) across the truncation window before the
optimizer step. It also exposes `grad_clip_norm` (default `None`), which
clips each accumulated weight-gradient's norm before the update — useful
since vanilla tanh RNNs are prone to exploding gradients over longer
sequences (Bengio et al. 1994; Pascanu et al. 2013).

Known limitation: `RNN.backward()` updates two differently-shaped weight
matrices (`FC_h`'s and `FC_y`'s) through one shared optimizer instance. A
*stateless* optimizer like `Sgd` is fine with that; a *stateful* one
(`SgdWithMomentum`, `Adam`) isn't — its momentum/second-moment accumulator
gets shaped after whichever matrix it saw last, so the next call's shape
mismatch throws. `demo_delayed_echo.py` uses plain `Sgd` for this reason.

## Small application: `demo_delayed_echo.py`

Trains the RNN to echo each input bit back 2 timesteps later — a task that
*requires* real short-term memory (a model that only sees the current input
can't beat chance, since the target isn't a function of it) but, unlike an
unbounded task such as running parity over the whole sequence, only asks the
hidden state to carry a few bits forward, which is tractable for a small
vanilla RNN trained with plain SGD in a short run.

```bash
cd Recurrent-Neural-Networks
python3 demo_delayed_echo.py
```

Typical result: held-out per-timestep accuracy in the 0.7–0.8 range against
a 0.5 random-guess baseline. Training is pure online SGD (one sequence → one
gradient step, no mini-batch averaging), so it's noisy — accuracy doesn't
rise monotonically even with a learning-rate decay schedule — which is why
the script tracks and prints the *best* checkpoint seen during training
alongside the final epoch, rather than only the last number.

## Usage (library, without the demo)

```bash
cd Recurrent-Neural-Networks
python3 -c "
from Layers.FullyConnected import FullyConnected
from Optimization.Optimizers import Sgd
import numpy as np

layer = FullyConnected(4, 3)
layer.optimizer = Sgd(0.01)
out = layer.forward(np.random.rand(2, 4))
print(out.shape)  # (2, 3)
"
```

There's no bundled dataset or unit-test harness beyond the demo — this is
the layer/optimizer library itself, ready to be driven by your own training
loop via `NeuralNetwork.py`.

## Requirements

```
numpy
scipy         # only needed by Layers/Conv.py (scipy.signal.correlate2d/convolve2d)
matplotlib    # optional, only for demo_delayed_echo.py's loss-curve plot
```
