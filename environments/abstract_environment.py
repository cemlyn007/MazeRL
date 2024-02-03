import abc
import numpy as np


class AbstractEnvironment(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool]:
        raise NotImplementedError

    def draw(self, state: np.ndarray) -> None:
        raise NotImplementedError
