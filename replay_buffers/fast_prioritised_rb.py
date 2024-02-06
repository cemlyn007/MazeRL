import torch
from numpy import ndarray

import abstract_agent
from replay_buffers import abstract_replay_buffer
import functools
import operator

class FastPrioritisedExperienceReplayBuffer(abstract_replay_buffer.ReplayBuffer):
    def __init__(self, max_capacity: int, batch_size: int, eps: float,
                 agent: abstract_agent.AbstractAgent, state_shape: tuple[int, ...]):
        self.batch_size = batch_size
        self._max_capacity = max_capacity
        self._index = 0
        self._full = False
        state_size = functools.reduce(operator.mul, state_shape, 1)
        self._container = torch.empty((max_capacity, state_size + 1 + 1 + state_size), dtype=torch.float32)
        self.eps = eps
        self.agent = agent
        self.dqn = agent.dqn
        self._weights = torch.zeros(max_capacity, dtype=torch.float32)
        self._indices = torch.empty((batch_size,), dtype=torch.int64)

    def store(self, state: ndarray, action: int, reward: float,
              new_state: ndarray):
        entry = torch.tensor((*state, action, reward, *new_state), dtype=torch.float32)
        self._container[self._index] = entry
        self._weights[self._index] = torch.max(self._weights) if self._index > 0 else 1.

        self._index += 1
        if self._index == self._max_capacity:
            self._full = True
            self._index = 0

    def batch_sample(self) -> torch.Tensor:
        if self._full:
            torch.multinomial(self._weights, self.batch_size, out=self._indices)
        else:
            torch.multinomial(self._weights[:self._index], self.batch_size, out=self._indices)
        return self._container[self._indices]

    def update_batch_weights(self, losses: torch.Tensor) -> None:
        batch_weights = torch.abs(losses) + self.eps
        self._weights[self._indices] = batch_weights

    def __len__(self):
        return self._max_capacity if self._full else self._index