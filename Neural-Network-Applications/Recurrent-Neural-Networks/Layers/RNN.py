import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.FC_h = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_y = FullyConnected(hidden_size, output_size)
        self.gradient_weights_n = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights_y = None
        self.weights_h = None
        self.weights = self.FC_h.weights
        self.bptt = 0
        self.h_t = None
        self.prev_h_t = None
        self.batch_size = None
        self.optimizer = None
        # Opt-in gradient-norm clipping (None = off, matching the original
        # behavior). Vanilla tanh RNNs trained with plain SGD over more than
        # a handful of BPTT steps are well known to be prone to exploding
        # gradients (Bengio et al. 1994; Pascanu et al. 2013) -- this repo
        # had no mechanism for it at all. Set e.g. `rnn.grad_clip_norm = 5.0`
        # to cap the global norm of each accumulated weight gradient before
        # the optimizer step.
        self.grad_clip_norm = None

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        if self._memorize:
            if self.h_t is None:
                self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))
            else:
                self.h_t[0] = self.prev_h_t
        else:
            self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))

        y_t = np.zeros((self.batch_size, self.output_size))

        # concatenating x,ht-1 and 1 to do forwarding to obtain new hidden state ht
        # 1: for t from 1 to T do:
        # 2:    ut = W hh · h t − 1 + W xh · x t + b h --> h t = tanh (x̃ t · W h )
        # 3:    h t = tanh ( u t )
        # 4:    o t = W hy · h t + b y
        # 5:    ŷ t = σ( o t )

        for b in range(self.batch_size):
            hidden_ax = self.h_t[b][np.newaxis, :]
            input_ax = input_tensor[b][np.newaxis, :]
            # x̃_t:
            input_new = np.concatenate((hidden_ax, input_ax), axis = 1)

            w_t = self.FC_h.forward(input_new)
            self.h_t[b+1] = TanH().forward(w_t) # h t = tanh (x̃ t · W h )
            y_t[b] = (self.FC_y.forward(self.h_t[b + 1][np.newaxis, :]))
        
        self.prev_h_t = self.h_t[-1]
        self.input_tensor = input_tensor

        return y_t

    def backward(self, error_tensor):

        self.out_error = np.zeros((self.batch_size, self.input_size))

        # Zero-initialized to match FC_y/FC_h's real weight shapes so BPTT can
        # accumulate into them below. (The original coursework export used
        # (hidden_size + 1, ...) / (hidden_size + input_size + 1, ...) here,
        # which assumes a bias-concatenated weight layout that FullyConnected
        # doesn't actually use -- it keeps bias as a separate parameter -- so
        # those shapes never matched FC_y/FC_h.gradient_weights. It went
        # unnoticed because the loop below used `=` instead of `+=`, which
        # silently replaced the wrong-shaped zeros every iteration anyway.)
        self.gradient_weights_y = np.zeros_like(self.FC_y.weights)
        self.gradient_weights_h = np.zeros_like(self.FC_h.weights)

        count = 0

        grad_tanh = 1-self.h_t[1::] ** 2
        hidden_error = np.zeros((1, self.hidden_size))

        # 1: for t from 1 to T do:
        # 2:    Run RNN for one step, computing h_t and y_t
        # 3:    if t mod k_1 == 0:
        # 4:        Run BPTT from t down to t-k_2
        
        for b in reversed(range(self.batch_size)):
            yh_error = self.FC_y.backward(error_tensor[b][np.newaxis, :])

            grad_yh = hidden_error + yh_error
            grad_hidden = grad_tanh[b]*grad_yh
            xh_error = self.FC_h.backward(grad_hidden)
            hidden_error = xh_error[:, 0:self.hidden_size]
            x_error = xh_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]
            self.out_error[b] = x_error

            if count <= self.bptt:
                self.weights_y = self.FC_y.weights
                self.weights_h = self.FC_h.weights
                # Sum (not overwrite) each timestep's local weight gradient --
                # gradient_weights_y/h are zero-initialized above specifically
                # so BPTT can accumulate across timesteps; a plain `=` here
                # (as in the original coursework export) silently discards
                # every timestep's contribution except the last one visited,
                # which starves the network of most of its training signal
                # and, in practice, destabilizes training on any task with a
                # dependency longer than one step (see demo_delayed_echo.py).
                self.gradient_weights_y = self.gradient_weights_y + self.FC_y.gradient_weights
                self.gradient_weights_h = self.gradient_weights_h + self.FC_h.gradient_weights
            count += 1

        if self.grad_clip_norm is not None:
            for grad in (self.gradient_weights_h, self.gradient_weights_y):
                norm = np.linalg.norm(grad)
                if norm > self.grad_clip_norm:
                    grad *= self.grad_clip_norm / (norm + 1e-8)

        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
            self.weights_h = self.optimizer.calculate_update(self.weights_h, self.gradient_weights_h)
            self.FC_y.weights = self.weights_y
            self.FC_h.weights = self.weights_h
        return self.out_error

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_y.initialize(weights_initializer, bias_initializer)
        self.FC_h.initialize(weights_initializer, bias_initializer)

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_n

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_y.gradient_weights = gradient_weights