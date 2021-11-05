from pathlib import Path

import cupy as cp

from .utils import to_gpu

try:
    with open(Path(__file__).parents[0] / "src/ompbatch.cu") as f:
        source = """
        extern "C" {{
            {}
        }}
        """
        source = source.format(f.read())
except FileNotFoundError:
    raise FileNotFoundError("ompbatch.cu not in src directory")

omp_batch_core = cp.RawModule(code=source).get_function("omp_batch")


class OmpBatch:
    """
    OmpBatch.

    Holds gpu address handles. Includes methods to convert to and from np arrays.
    """

    def __init__(self, n_atoms, sparsity_target, batch_size, epsilon=1e-12):
        """
        Initializer for OmpBatch class.

        Args:
            n_atoms (int): Total no of dictionary atoms.
            sparsity_target (int): Target sparsity.
                sparsity_target < n_atoms for useful learning.
            batch_size (int): Batches to run in parallel.
            epsilon (float): Small scalar for numerical stability.
        """
        self.n_atoms = cp.int32(n_atoms)
        self.sparsity_target = cp.int32(sparsity_target)
        self.batch_size = batch_size
        self.epsilon = cp.float32(epsilon)

        self.L_gpu = cp.zeros((batch_size, sparsity_target, sparsity_target), dtype=cp.float32)
        self.I_gpu = cp.zeros((batch_size, sparsity_target), dtype=cp.float32)
        self.w_gpu = cp.zeros((batch_size, sparsity_target), dtype=cp.float32)

        self.a_gpu = cp.zeros((batch_size, n_atoms), dtype=cp.float32)
        self.gamma_gpu = cp.zeros((batch_size, n_atoms), dtype=cp.float32)

        self.n_bytes = batch_size * (sparsity_target * (2 + sparsity_target) + 3 * n_atoms) + n_atoms ** 2

    def omp_batch(self, a_0, gram, as_gpu=False):
        """
        Omp batch algorithm.

        Args:
            a_0 (np.ndarray | cp.ndarray): Set of vectors [D^T @ X].
            gram (np.ndarray | cp.ndarray): Gram matrix [D^T @ D].
            as_gpu (bool): Get output as cp.ndarray.
        Returns:
            np.ndarray: Sparse encodings.
        """
        a_0 = cp.asfortranarray(to_gpu(a_0))
        gram = to_gpu(gram)
        omp_batch_core(
            (self.batch_size,),
            (1,),
            (
                a_0,
                gram,
                self.sparsity_target,
                self.n_atoms,
                self.I_gpu,
                self.L_gpu,
                self.w_gpu,
                self.a_gpu,
                self.gamma_gpu,
                self.epsilon,
            ),
        )
        if as_gpu:
            return self.gamma_gpu
        return cp.asnumpy(self.gamma_gpu).T
