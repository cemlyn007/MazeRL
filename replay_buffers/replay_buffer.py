import collections
import random

import numpy as np
import torch
import torch.utils.data


class ReplayBuffer(torch.utils.data.Dataset):

    def __init__(self, max_capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.container = collections.deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.container)

    def store(self, state: np.ndarray, action: int, reward: float,
              new_state: np.ndarray):
        entry = torch.tensor((*state, action, reward, *new_state))
        self.container.append(entry)

    def sample(self) -> torch.Tensor:
        return random.choice(self.container)

    def batch_sample(self) -> torch.Tensor:
        transitions = random.sample(self.container, self.batch_size)
        return torch.stack(transitions)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.container[index]

    def __add__(self, other):
        raise NotImplementedError
