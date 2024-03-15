import torch
import jax
import abc
from typing import TypeVar, Generic

T = TypeVar('T', torch.Tensor, jax.Array)

class AbstractDQN(Generic[T], abc.ABC):
    @abc.abstractproperty
    def has_target_network(self) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def predict_q_values(self, observations: T) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def train_q_network(self, transition: T) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_losses(self, transitions: T) -> T:
        raise NotImplementedError
