import torch
import abc

class AbstractDQN(abc.ABC):
    @abc.abstractproperty
    def has_target_network(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def train_q_network(self, transition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
