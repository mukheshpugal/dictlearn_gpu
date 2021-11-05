# GPU Dictionary learning for python

[Cuda](https://en.wikipedia.org/wiki/CUDA) implementations of dictionary learning algorithms. This project was inspired by the [dictlearn](https://github.com/permfl/dictlearn) python library.

## Install
1. This package requires Python 3+ installed. You can install it from the official website (https://www.python.org/downloads/)
2. This package also requires `nvcc` to be installed and requires it to be in `PATH`.
3. Install this package with `python3 -m pip install git+https://github.com/mukheshpugal/dictlearn_gpu.git@master`

## Usage
Use this module as follows:

### High level API
#### Train a dictionary
```py
from dictlearn_gpu import train_dict
from dictlearn_gpu.utils import dct_dict_1d

signals = ... # Load signals. Shape == (L, N)

dictionary = dct_dict_1d(
    n_atoms=32,
    size=L,
)

new_dictionary, errors, iters = train_dict(signals, dictionary, sparsity_target=8)
```

### Low level API
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

#### MOD
```py
from dictlearn_gpu import update_dict_mod

signals = ... # Load signals
decomp = ... # Sparse decomposition

updated_dictionary = update_dict_mod(signals, decomp)
```
