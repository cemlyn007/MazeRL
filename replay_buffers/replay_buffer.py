import numpy as np
import torch
from collections import deque
from torch.utils.data import Dataset
import random
from numpy import ndarray


class ReplayBuffer(Dataset):

    def __init__(self, max_capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.container = deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.container)

    def store(self, state: ndarray, action: int, reward: float,
              new_state: ndarray):
        entry = torch.tensor((*state, action, reward, *new_state))
        self.container.append(entry)

    def sample(self):
        random_sample_index = np.random.randint(0, len(self))
        return self[random_sample_index]

    def batch_sample(self):
        transitions = random.sample(self.container, self.batch_size)
        return torch.stack(transitions)

    def __getitem__(self, index: int):
        return self.container[index]

    def __add__(self, other):
        raise NotImplementedError
