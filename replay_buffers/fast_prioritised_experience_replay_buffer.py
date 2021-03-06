import random
from collections import deque

import torch
from numpy import ndarray

from abstract_agent import AbstractAgent
from replay_buffers import ReplayBuffer


class FastPrioritisedExperienceReplayBuffer(ReplayBuffer):

    def __init__(self, max_capacity: int, batch_size: int, eps: float,
                 agent: AbstractAgent):
        super().__init__(max_capacity, batch_size)
        self.eps = eps
        self.agent = agent
        self.dqn = agent.dqn
        self.weights = deque(maxlen=max_capacity)
        self._indices = None

    def store(self, state: ndarray, action: int, reward: float,
              new_state: ndarray):
        entry = torch.tensor((*state, action, reward, *new_state))
        self.container.append(entry)
        weight = max(self.weights) if len(self.weights) > 0 else 1.
        self.weights.append(weight)

    def sample(self):
        transition = random.choices(self.container, self.weights)
        return transition

    def batch_sample(self):
        self._indices = random.choices(range(len(self.container)),
                                       self.weights, k=self.batch_size)
        transitions = [self.container[i] for i in self._indices]
        return torch.stack(transitions)

    def update_batch_weights(self, losses: torch.Tensor):
        batch_weights = torch.abs(losses) + self.eps
        for index, weight in zip(self._indices, batch_weights):
            self.weights[index] = weight.item()

    def __add__(self, other):
        raise NotImplementedError
