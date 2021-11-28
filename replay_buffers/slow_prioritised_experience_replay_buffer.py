import random
import collections

import numpy as np
import torch

import abstract_agent
from replay_buffers import replay_buffer


class SlowPrioritisedExperienceReplayBuffer(replay_buffer.ReplayBuffer):

    def __init__(self, max_capacity: int, batch_size: int, eps: float,
                 alpha: float, agent: abstract_agent.AbstractAgent):
        super().__init__(max_capacity, batch_size)
        self.weights = collections.deque(maxlen=max_capacity)
        self.eps = eps
        self.alpha = alpha
        self.agent = agent
        self.dqn = agent.dqn

    def weight(self, entry: torch.Tensor):
        with torch.no_grad():
            entry = entry.unsqueeze(0).to(self.dqn.device)
            loss = self.dqn.compute_losses(entry).item()
        return abs(loss) + self.eps

    def store(self, state: np.ndarray, action: int, reward: float,
              new_state: np.ndarray):
        entry = torch.tensor((*state, action, reward, *new_state))
        self.container.append(entry)
        weight = self.weight(entry)
        self.weights.append(weight)

    def sample(self):
        weights = self.get_sampling_weights()
        transition = random.choices(self.container, weights)
        return transition

    def get_sampling_weights(self):
        if self.alpha == 1:
            return self.weights
        else:
            ps = np.array(self.weights) ** self.alpha
            ps /= ps.sum()
            return ps

    def batch_sample(self):
        weights = self.get_sampling_weights()
        transitions = random.choices(self.container, weights, k=self.batch_size)
        return torch.stack(transitions)

    def __add__(self, other):
        raise NotImplementedError
