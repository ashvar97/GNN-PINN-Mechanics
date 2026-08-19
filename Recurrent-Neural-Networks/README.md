# Recurrent Neural Networks

A from-scratch NumPy deep learning framework (no PyTorch/TensorFlow) covering
fully-connected, convolutional, and recurrent architectures. This folder
consolidates a 3-part coursework exercise series
(FC layers → CNN → RNN/LSTM, each building on the last) into one package:
the earlier `Convolution-Neural-Network-main` and `Fully-connected-Neural-Networks-main`
snapshots were strictly superseded by this one (verified file-by-file — every
shared file here is identical to or a superset of theirs), so this is the
single, complete, up-to-date version.

## What changed from the original coursework export

The original three folders all imported from `Layers`/`Optimization`
packages (`from Layers import Base`, `from Optimization import Optimizers`,
...) that didn't actually exist on disk — files just sat flat, so none of
them could run as extracted. This folder fixes that by physically organizing
the code into real packages matching those imports:

```
Recurrent-Neural-Networks/
├── NeuralNetwork.py        # composes layers into a trainable network
├── Layers/
│   ├── Base.py              # BaseLayer: shared layer interface
│   ├── FullyConnected.py
│   ├── Conv.py               # 2D convolution (im2col via scipy.signal)
│   ├── Pooling.py
│   ├── Flatten.py
│   ├── ReLU.py / SoftMax.py / Sigmoid.py / TanH.py
│   ├── Dropout.py
│   ├── BatchNormalization.py
│   ├── RNN.py / LSTM.py
│   └── Helpers.py            # see note below
└── Optimization/
    ├── Optimizers.py         # SGD, SGD+Momentum, Adam (+ regularizer hook)
    ├── Loss.py                # CrossEntropy
    ├── Constraints.py         # L1/L2 weight regularizers
    └── Initializers.py       # Constant, UniformRandom, Xavier, He
```

`Layers/Helpers.py` did not exist in any of the three exported folders even
though `BatchNormalization.py` imports `compute_bn_gradients` from it — it
was presumably course-internal grading scaffolding never included in these
exports. It's been reimplemented from scratch here using the standard
analytic batch-norm backward-pass formula (Ioffe & Szegedy, 2015); see the
docstring in that file.

No other logic was changed — layer implementations are untouched, only
relocated.

## Usage

Run scripts from inside this folder (or add it to `PYTHONPATH`) so the
`Layers`/`Optimization` package imports resolve:

```bash
cd Recurrent-Neural-Networks
python3 -c "
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Optimization.Optimizers import Sgd
import numpy as np

layer = FullyConnected(4, 3)
layer.optimizer = Sgd(0.01)
out = layer.forward(np.random.rand(2, 4))
print(out.shape)  # (2, 3)
"
```

There's no bundled dataset/training script or unit-test harness (the
original course's grading tests were never part of these exports) — this is
the layer/optimizer library itself, ready to be driven by your own training
loop via `NeuralNetwork.py`.

## Requirements

```
numpy
scipy   # only needed by Layers/Conv.py (scipy.signal.correlate2d/convolve2d)
```
