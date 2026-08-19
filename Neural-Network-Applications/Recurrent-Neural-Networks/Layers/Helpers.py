"""
`BatchNormalization.py` calls `Helpers.compute_bn_gradients(...)`, but no
`Helpers.py` was present in any of the three exported coursework folders --
it was presumably part of the course's private grading scaffolding and never
included in these submission-only exports.

This is a from-scratch reimplementation of the one function that's actually
called (`compute_bn_gradients`), using the standard analytic backward-pass
derivation for batch normalization (Ioffe & Szegedy, 2015, "Batch
Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift" -- the vectorized gradient-w.r.t.-input formula from their
appendix, also widely reproduced in public backprop tutorials). It is written
independently here, not copied from the original course material (which was
never available to write from).
"""
from __future__ import annotations

import numpy as np


def compute_bn_gradients(error_tensor: np.ndarray, input_tensor: np.ndarray,
                          weights: np.ndarray, mean: np.ndarray, var: np.ndarray,
                          eps: float = np.finfo(float).eps) -> np.ndarray:
    """Gradient of the batch-norm loss w.r.t. its input (dX).

    `dgamma`/`dbeta` are NOT computed here -- BatchNormalization.backward()
    already computes those directly from `error_tensor * X_hat` and
    `error_tensor`, so this only needs to return dX.

    Parameters
    ----------
    error_tensor : dL/dY, shape (N, C)
    input_tensor : the layer's input X from the forward pass, shape (N, C)
    weights      : gamma (scale), shape (C,)
    mean, var    : batch mean/variance used in the forward pass, shape (C,)
    """
    N = input_tensor.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)
    x_mu = input_tensor - mean

    dx_hat = error_tensor * weights
    dvar = np.sum(dx_hat * x_mu, axis=0) * -0.5 * std_inv ** 3
    dmean = np.sum(dx_hat * -std_inv, axis=0) + dvar * np.mean(-2.0 * x_mu, axis=0)

    dx = dx_hat * std_inv + dvar * 2.0 * x_mu / N + dmean / N
    return dx
