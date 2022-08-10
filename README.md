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

## Performance analysis
Functionalities of this module were tested against equivalents from the python libraries [dictlearn](https://github.com/permfl/dictlearn) and [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning). The following results are obtained:

### Training on 1 dimensional signals
|Training time|Training loss|
|--|--|
|![1D train time](https://user-images.githubusercontent.com/39578914/183962460-7c993fae-2ebd-4b7f-b1ce-e47ea5a729fb.png)|![1D train loss](https://user-images.githubusercontent.com/39578914/183962485-6068ac53-b1b2-4fb7-8ccc-8551eab5da0e.png)|

### Training on 2 dimensional signals
|Training time|Training loss|
|--|--|
|![2D train time](https://user-images.githubusercontent.com/39578914/183963666-6869e198-520f-4fc2-a689-bff87c862d52.png)|![Screenshot (8)](https://user-images.githubusercontent.com/39578914/183963781-2408265e-94ab-4776-857b-a1f8d374a70e.png)|
