import numpy as np


def trunc(values: np.ndarray, decs: int = 0) -> np.ndarray:
    return (np.trunc(values * 10 ** decs) / (10 ** decs)).astype(values.dtype)
