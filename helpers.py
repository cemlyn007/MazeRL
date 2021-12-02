import typing

import numpy as np


def trunc(values: np.ndarray, decs: int = 0) -> np.ndarray:
    return (np.trunc(values * 10 ** decs) / (10 ** decs)).astype(values.dtype)


class Hyperparameters(typing.NamedTuple):
    gamma: float = 0.99
    lr: float = 5.e-4
    weight_decay: float = 1.e-7
