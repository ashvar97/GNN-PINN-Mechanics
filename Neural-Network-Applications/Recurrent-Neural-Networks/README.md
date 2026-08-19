# Recurrent Neural Networks

A from-scratch NumPy deep learning framework (no PyTorch/TensorFlow) covering
fully-connected, convolutional, and recurrent architectures, plus a small
working application (`demo_delayed_echo.py`) that actually trains the RNN
on a real task rather than leaving it as unexercised library code.

This folder consolidates a 3-part coursework exercise series (FC layers →
CNN → RNN/LSTM, each building on the last) into one package: the sibling
[`../Convolution-Neural-Network`](../Convolution-Neural-Network) and
[`../Fully-connected-Neural-Networks`](../Fully-connected-Neural-Networks)
snapshots were strictly superseded by this one (verified file-by-file —
every shared file here is identical to or a superset of theirs), so this is
the single, complete, up-to-date version; the other two are kept only as
untouched historical snapshots of the earlier exercises.

## Structure

The original three folders all imported from `Layers`/`Optimization`
packages (`from Layers import Base`, `from Optimization import Optimizers`,
...) that didn't actually exist on disk — files just sat flat, so none of
them could run as extracted. This folder fixes that by physically organizing
the code into real packages matching those imports:

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

`Layers/Helpers.py` did not exist in any of the three exported folders even
though `BatchNormalization.py` imports `compute_bn_gradients` from it — it
was presumably course-internal grading scaffolding never included in these
exports. It's been reimplemented from scratch here using the standard
analytic batch-norm backward-pass formula (Ioffe & Szegedy, 2015); see the
docstring in that file.

Dead code found while reading through every file was removed: a duplicate
commented-out `TanH` class, stray dead comments in `Sigmoid.py`/`Pooling.py`,
an unused `self.temp` attribute in `FullyConnected.py`, and a write-only
`self.h_mem` list in `RNN.py` that was appended to but never read.

## The RNN actually needed two real bug fixes to train at all

Wiring up `demo_delayed_echo.py` (below) surfaced genuine correctness bugs
in `Layers/RNN.py`, not just style issues — both are now fixed there, with
inline comments explaining each:

1. **Truncated-BPTT gradient accumulation used `=` instead of `+=`.**
   `gradient_weights_y`/`gradient_weights_h` are zero-initialized right
   before the backward loop (clearly meant to accumulate), but the loop body
   overwrote them each iteration. Combined with `bptt` defaulting to `0`,
   this meant only one timestep's local gradient ever reached the weight
   update — effectively no backprop-through-time at all.
2. **The zero-init shapes were wrong.** They assumed a bias-concatenated
   weight layout (`hidden_size + 1`, `hidden_size + input_size + 1`) that
   `FullyConnected` doesn't use — it keeps bias as a separate parameter — so
   they never actually matched `FC_y`/`FC_h.gradient_weights`'s real shape.
   This went unnoticed because bug #1's `=` silently replaced the
   wrong-shaped zeros every step anyway; fixing #1 first surfaced #2 as a
   `ValueError` immediately.

Two harmless dead-code lines were also removed from `RNN.backward()`: it
wrote to `self.FC_y.input_tensor`/`self.FC_h.input_tensor`, an attribute
`FullyConnected` never reads (only `.lastIn` matters to it).

Even after both fixes, plain SGD with full BPTT still diverged to NaN after
a few hundred updates — the well-documented exploding-gradient failure mode
of vanilla tanh RNNs (Bengio et al. 1994; Pascanu et al. 2013), which this
repo had no mitigation for. Added `RNN.grad_clip_norm` (default `None`, so
existing behavior is unchanged unless you opt in) — set it to clip each
accumulated weight-gradient's norm before the optimizer step.

One more real limitation, left as-is rather than "fixed": `RNN.backward()`
updates two differently-shaped weight matrices (`FC_h`'s and `FC_y`'s)
through one shared optimizer instance. A *stateless* optimizer like `Sgd`
is fine with that; a *stateful* one (`SgdWithMomentum`, `Adam`) isn't — its
momentum/second-moment accumulator gets shaped after whichever matrix it
saw last, so the next call's shape mismatch throws. Giving `FC_h`/`FC_y`
their own optimizer instances would fix this properly but touches more of
`RNN.py`'s update logic than this pass was scoped for, so `demo_delayed_echo.py`
deliberately uses plain `Sgd`.

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

There's no bundled dataset or unit-test harness beyond the demo (the
original course's grading tests were never part of these exports) — this is
the layer/optimizer library itself, ready to be driven by your own training
loop via `NeuralNetwork.py`.

## Requirements

```
numpy
scipy         # only needed by Layers/Conv.py (scipy.signal.correlate2d/convolve2d)
matplotlib    # optional, only for demo_delayed_echo.py's loss-curve plot
```
