import random
from collections import deque

import torch
from numpy import ndarray

import abstract_agent
from replay_buffers import replay_buffer


class FastPrioritisedExperienceReplayBuffer(replay_buffer.ReplayBuffer):

    def __init__(self, max_capacity: int, batch_size: int, eps: float,
                 agent: abstract_agent.AbstractAgent):
        super().__init__(max_capacity, batch_size)
        self.eps = eps
        self.agent = agent
        self.dqn = agent.dqn
        self.weights = deque(maxlen=max_capacity)
        self._indices = None

    def store(self, state: ndarray, action: int, reward: float,
              new_state: ndarray):
        entry = torch.tensor((*state, action, reward, *new_state), dtype=torch.float32)
        self.container.append(entry)
        weight = max(self.weights) if len(self.weights) > 0 else 1.
        self.weights.append(weight)

    def batch_sample(self) -> torch.Tensor:
        self._indices = random.choices(range(len(self.container)),
                                       self.weights, k=self.batch_size)
        transitions = [self.container[i] for i in self._indices]
        return torch.stack(transitions)

    def update_batch_weights(self, losses: torch.Tensor) -> None:
        batch_weights = torch.abs(losses) + self.eps
        for index, weight in zip(self._indices, batch_weights):
            self.weights[index] = weight.item()
