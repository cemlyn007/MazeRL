import abc
import numpy as np
import torch


class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def store(self, state: np.ndarray, action: int, reward: float,
              done: bool, new_state: np.ndarray):
        pass

    @abc.abstractmethod
    def batch_sample(self) -> torch.Tensor:
        pass
