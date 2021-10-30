# GPU Dictionary learning for python

[Cuda](https://en.wikipedia.org/wiki/CUDA) implementations of dictionary learning algorithms. This project was inspired by the [dictlearn](https://github.com/permfl/dictlearn) python library.

## Usage
Use this module as follows:

#### OMP-batch
```py
import numpy as np

from dictlearn_gpu import OmpBatch
from dictlearn_gpu.utils import dct_dict_1d

signals = ... # Load signals

dictionary = dct_dict_1d(
    n_atoms=32,
    size=16,
)
ob = OmpBatch(n_atoms=32, sparsity_target=8, batch_size=2048)
decomp = ob.omp_batch(a_0=dictionary.T @ signals, gram=dictionary.T @ dictionary)

print(np.sum((signals - dictionary.dot(decomp)) ** 2))  # Reconstruction error
```