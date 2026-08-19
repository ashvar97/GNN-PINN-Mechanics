"""
A small, self-contained use of this from-scratch RNN: learn a delayed echo.

Given a sequence of random bits x_1..x_T, predict at every step t the bit
that appeared `DELAY` steps earlier (x_{t-DELAY}). This needs genuine
short-term memory -- a model that only ever sees the current input cannot
beat chance, since the correct answer isn't a function of x_t at all -- but
unlike an unbounded task (e.g. running parity over the whole sequence), a
fixed short delay only asks the hidden state to carry a few bits forward,
which is something a small vanilla RNN can plausibly learn in a short
training run.

Pipeline: RNN -> SoftMax -> CrossEntropyLoss (all straight from Layers/ and
Optimization/, nothing new). Getting this to learn at all surfaced two real
bugs in Layers/RNN.py, now fixed there (see its comments for details):

  1. Truncated-BPTT gradient accumulation used `=` instead of `+=`, so only
     one timestep's local gradient ever reached the weight update by
     default (`bptt` defaults to 0).
  2. The zero-init shapes for the accumulated gradients assumed a
     bias-concatenated weight layout FullyConnected doesn't use, so they
     silently didn't match FC_h/FC_y.gradient_weights's real shape -- masked
     as long as bug #1's `=` kept discarding them every step.

Even after both fixes, plain SGD over multiple BPTT steps still diverged to
NaN after a few hundred updates -- the well-known exploding-gradient failure
mode of vanilla tanh RNNs (Bengio et al. 1994; Pascanu et al. 2013), which
this repo had no mitigation for at all. `RNN.grad_clip_norm` is a new
opt-in attribute (default `None`, i.e. no behavior change unless set) that
clips each accumulated weight-gradient's norm before the optimizer step.

A shared optimizer instance updates two differently-shaped weight matrices
inside RNN.backward() (FC_h's and FC_y's), so a *stateful* optimizer
(SgdWithMomentum, Adam) doesn't work here at all -- their momentum/
second-moment accumulators are shaped after whichever matrix they saw last,
so the next call's shape mismatch throws. Plain Sgd is used deliberately.

Training here is pure online SGD (one sequence -> one gradient step, no
mini-batch averaging), which is noisy -- held-out accuracy doesn't rise
monotonically. A simple step-decay learning-rate schedule (halve every
`LR_DECAY_EVERY` epochs) tames most of that late-training noise, but this
script still tracks and reports the best checkpoint seen during training
rather than just the final epoch, and prints the final epoch's result too,
for full transparency about what's actually happening.

Run:
    cd Recurrent-Neural-Networks
    python3 demo_delayed_echo.py
"""
from __future__ import annotations

import numpy as np

from Layers.RNN import RNN
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Sgd
from Optimization.Initializers import Xavier, Constant

SEQ_LEN = 10
DELAY = 2
HIDDEN_SIZE = 16
LEARNING_RATE = 0.08
LR_DECAY_EVERY = 50    # halve the learning rate every this many epochs
LR_DECAY_FACTOR = 0.5
GRAD_CLIP_NORM = 1.0
EPOCHS = 200
SEQUENCES_PER_EPOCH = 64
EVAL_EVERY = 20
TEST_SEQUENCES = 300
SEED = 0


def make_sequence(seq_len: int, delay: int, rng: np.random.Generator):
    """One random bit sequence + its `delay`-step-echoed one-hot targets.
    The first `delay` steps have no valid target (nothing to echo yet) and
    are excluded from loss/accuracy via EVAL from `delay` onward."""
    bits = rng.integers(0, 2, size=seq_len).astype(np.float32)
    target = np.zeros(seq_len, dtype=int)
    target[delay:] = bits[: seq_len - delay].astype(int)
    X = bits.reshape(seq_len, 1)
    Y = np.zeros((seq_len, 2), dtype=np.float32)
    Y[np.arange(seq_len), target] = 1.0
    return X, Y


def step_accuracy(probs: np.ndarray, Y: np.ndarray, delay: int) -> float:
    """Accuracy over steps >= delay only (steps before that have no real target)."""
    pred = np.argmax(probs[delay:], axis=1)
    true = np.argmax(Y[delay:], axis=1)
    return float(np.mean(pred == true))


def evaluate(rnn: RNN, softmax: SoftMax, seq_len: int, delay: int, n_sequences: int, rng):
    accs = []
    for _ in range(n_sequences):
        X, Y = make_sequence(seq_len, delay, rng)
        probs = softmax.forward(rnn.forward(X))
        accs.append(step_accuracy(probs, Y, delay))
    return float(np.mean(accs))


def main():
    # Optimization/Initializers.py draws from NumPy's *global* random state
    # (plain np.random.randn/uniform, not a seeded Generator), unlike the
    # rest of this script which uses np.random.default_rng() streams -- so
    # this call is required for weight initialization to actually be
    # reproducible run-to-run.
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    rnn = RNN(input_size=1, hidden_size=HIDDEN_SIZE, output_size=2)
    rnn.memorize = False            # each sequence starts from a zero hidden state
    rnn.bptt = SEQ_LEN               # backprop through the whole sequence, not just 1 step
    rnn.grad_clip_norm = GRAD_CLIP_NORM
    lr = LEARNING_RATE
    rnn.optimizer = Sgd(lr)
    rnn.initialize(Xavier(), Constant(0.0))
    softmax = SoftMax()
    loss_fn = CrossEntropyLoss()

    test_rng = np.random.default_rng(SEED + 1)  # separate stream, held out from training

    print(f"Training RNN(1 -> {HIDDEN_SIZE} -> 2) to echo each bit back "
          f"{DELAY} steps later (sequence length {SEQ_LEN})...")
    loss_history = []
    best_acc, best_epoch = -1.0, -1
    for epoch in range(EPOCHS):
        if epoch > 0 and epoch % LR_DECAY_EVERY == 0:
            lr *= LR_DECAY_FACTOR
            rnn.optimizer = Sgd(lr)
        epoch_loss = 0.0
        for _ in range(SEQUENCES_PER_EPOCH):
            X, Y = make_sequence(SEQ_LEN, DELAY, rng)  # fresh sequences every epoch
            probs = softmax.forward(rnn.forward(X))
            epoch_loss += loss_fn.forward(probs, Y)
            grad = loss_fn.backward(Y)
            grad = softmax.backward(grad)
            rnn.backward(grad)
        loss_history.append(epoch_loss / SEQUENCES_PER_EPOCH)

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            test_acc = evaluate(rnn, softmax, SEQ_LEN, DELAY, TEST_SEQUENCES, test_rng)
            if test_acc > best_acc:
                best_acc, best_epoch = test_acc, epoch
            print(f"  epoch {epoch:3d}  train_loss={loss_history[-1]:.4f}  "
                  f"held_out_step_accuracy={test_acc:.3f}")

    final_acc = evaluate(rnn, softmax, SEQ_LEN, DELAY, TEST_SEQUENCES, test_rng)
    print(f"\nBest held-out accuracy: {best_acc:.3f} (epoch {best_epoch})")
    print(f"Final-epoch held-out accuracy: {final_acc:.3f}")
    print("Random-guess baseline: 0.500")
    print("(Online per-sequence SGD is noisy here -- accuracy doesn't rise "
          "monotonically -- which is why both the best checkpoint and the "
          "final epoch are reported rather than just the last number.)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(loss_history)
        plt.xlabel("epoch"); plt.ylabel("mean cross-entropy loss")
        plt.title(f"RNN learning {DELAY}-step delayed echo")
        plt.tight_layout()
        plt.savefig("demo_delayed_echo_loss.png")
        print("Saved loss curve to demo_delayed_echo_loss.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
