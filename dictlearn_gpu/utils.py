import numpy as np


def dct_dict_1d(n_atoms, size):
    dct = np.zeros((size, n_atoms))

    for k in range(n_atoms):
        basis = np.cos(np.arange(size) * k * np.pi / n_atoms)
        if k > 0:
            basis = basis - np.mean(basis)

        basis /= np.linalg.norm(basis)
        dct[:, k] = basis
    return dct
