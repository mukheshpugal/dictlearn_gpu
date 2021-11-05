import cupy as cp

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
