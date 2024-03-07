import collections
import random

import numpy as np
import jax.numpy as jnp
import jax
from jax_replay_buffers import abstract_replay_buffer


class ReplayBuffer(abstract_replay_buffer.ReplayBuffer):

    def __init__(self, max_capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.container = collections.deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.container)

    def store(self, state: np.ndarray, action: int, reward: float, done: bool,
              new_state: np.ndarray):
        entry = jnp.array((*state, action, reward, done, *new_state))
        self.container.append(entry)

    def batch_sample(self) -> jax.Array:
        transitions = random.sample(self.container, self.batch_size)
        return jnp.stack(transitions)
