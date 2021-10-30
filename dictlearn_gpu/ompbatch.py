from pathlib import Path

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

try:
    with open(Path(__file__).parents[0] / "src/ompbatch.cu") as f:
        source = f.read()
except FileNotFoundError:
    raise FileNotFoundError("ompbatch.cu not in src directory")

omp_batch_core = SourceModule(source).get_function("omp_batch")


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
        self.n_atoms = np.int32(n_atoms)
        self.sparsity_target = np.int32(sparsity_target)
        self.batch_size = batch_size
        self.epsilon = np.float32(epsilon)

        self.L_gpu = cuda.mem_alloc(batch_size * sparsity_target ** 2 * 4)
        self.I_gpu = cuda.mem_alloc(batch_size * sparsity_target * 4)
        self.w_gpu = cuda.mem_alloc(batch_size * sparsity_target * 4)

        self.a_gpu = cuda.mem_alloc(batch_size * n_atoms * 4)
        self.a_0_gpu = cuda.mem_alloc(batch_size * n_atoms * 4)
        self.gamma_gpu = cuda.mem_alloc(batch_size * n_atoms * 4)

        self.gamma = np.zeros(batch_size * n_atoms, dtype=np.float32)

        self.gram_gpu = cuda.mem_alloc(n_atoms ** 2 * 4)

        self.n_bytes = batch_size * (sparsity_target * (2 + sparsity_target) + 3 * n_atoms) + n_atoms ** 2

    def omp_batch(self, a_0, gram):
        """
        Omp batch algorithm.

        Args:
            a_0 (np.ndarray): Set of vectors [D^T @ X].
            gram (np.ndarray): Gram matrix [D^T @ D].
        Returns:
            np.ndarray: Sparse encodings.
        """
        cuda.memcpy_htod(self.a_0_gpu, a_0.astype(np.float32).transpose().reshape(-1))
        cuda.memcpy_htod(self.gram_gpu, gram.astype(np.float32))
        omp_batch_core(
            self.a_0_gpu,
            self.gram_gpu,
            self.sparsity_target,
            self.n_atoms,
            self.I_gpu,
            self.L_gpu,
            self.w_gpu,
            self.a_gpu,
            self.gamma_gpu,
            self.epsilon,
            block=(1, 1, 1),
            grid=(self.batch_size, 1, 1),
        )
        cuda.memcpy_dtoh(self.gamma, self.gamma_gpu)
        return self.gamma.reshape(self.batch_size, -1).T
