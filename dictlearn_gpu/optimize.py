import cupy as cp
import numpy as np

from .ompbatch import OmpBatch
from .utils import to_gpu


def update_dict_mod(signals, sparse, as_gpu=False, epsilon=1e-12):
    """Update dictionary using MOD.

    Args:
        signals (np.ndarray | cp.ndarray): Signals (or) training data.
        sparse (np.ndarray | cp.ndarray): Sparse codes.
        as_gpu (bool): Get output as cp.ndarray.
        epsilon (float): Small scalar for numerical stability.

    Returns:
        np.ndarray | cp.ndarray: New dictionary.
    """
    signals = to_gpu(signals)
    sparse = to_gpu(sparse)
    new = cp.dot(signals, cp.linalg.pinv(sparse))
    new /= cp.linalg.norm(new, axis=0) + epsilon
    if as_gpu:
        return new
    return cp.asnumpy(new)


def train_dict(
    signals, dict_init, sparsity_target, min_error=1e-3, max_iters=100, callbacks=None, verbose=0, as_gpu=False
):
    """Learn the optimal dictionary over several iterations.

    Args:
        signals (np.ndarray | cp.ndarray): Signals to learn on.
        dict_init (np.ndarray | cp.ndarray): Initial dictionary.
        sparsity_target (int): Target sparsity.
        min_error (float): Minimum error.
        max_iters (int): Maximum iterations.
        callbacks (list): List of functions to call once every iteration.
        verbose (int): Verbosity level.
        as_gpu (bool): Get output as cp.ndarray.

    Returns:
        np.ndarray | cp.ndarray: Final state of dictionary.
        list: List of errors.
        int: No. of total iterations.
    """
    batch_size = signals.shape[1]
    n_atoms = dict_init.shape[1]
    error_factor = 1 / np.sqrt(np.prod(signals.shape))
    if callbacks is None:
        callbacks = []

    signals = to_gpu(signals)
    dict_state = to_gpu(dict_init)

    ob = OmpBatch(n_atoms, sparsity_target, batch_size)

    errors = []
    for iter in range(1, max_iters + 1):
        # Sparse decomposition
        decomp = ob.omp_batch(dict_state.T @ signals, dict_state.T @ dict_state, as_gpu=True)

        # Update dictionary
        dict_state = update_dict_mod(signals, decomp, as_gpu=True)

        error = cp.linalg.norm(signals - dict_state @ decomp) * error_factor
        errors.append(error)

        # Run callbacks
        for callback in callbacks:
            callback(dict_state, signals, error)

        # Print stats
        if verbose >= 1:
            print(f"Iter: {iter}; Error: {error}")

        # Exit if required error reached
        if error < min_error:
            break

    if not as_gpu:
        dict_state = cp.asnumpy(dict_state)
    return dict_state, errors, iter
