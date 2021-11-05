# GPU Dictionary learning for python

[Cuda](https://en.wikipedia.org/wiki/CUDA) implementations of dictionary learning algorithms. This project was inspired by the [dictlearn](https://github.com/permfl/dictlearn) python library.

## Setup
- Install required packages with `pip install -r requirements.txt`
- Install appropriate version of [cupy](https://docs.cupy.dev/en/stable/install.html#installing-cupy) based on the installed version of CUDA.
- Requires `nvcc` to be in PATH.

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
