import numpy as np
import jax

from jax_replay_buffers import abstract_replay_buffer
import functools
import operator
import jax.numpy as jnp


class FastPrioritisedExperienceReplayBuffer(abstract_replay_buffer.ReplayBuffer):
    def __init__(self, max_capacity: int, batch_size: int, eps: float, state_shape: tuple[int, ...]):
        self.batch_size = batch_size
        self._max_capacity = max_capacity
        self._index = jnp.array(0, dtype=jnp.uint32)
        self._full = False
        state_size = functools.reduce(operator.mul, state_shape, 1)
        self._container = jnp.empty((max_capacity, state_size + 1 + 1 + 1 + state_size), dtype=jnp.float32)
        self.eps = eps
        self._weights = jnp.zeros(max_capacity, dtype=jnp.float32)
        self._indices = jnp.empty((batch_size,), dtype=jnp.uint32)
        self._key = jax.random.PRNGKey(0)
        self._store = jax.jit(self._store, donate_argnums=(0, 1, 2))
        self._jitted_update_batch_weights = jax.jit(self._update_batch_weights, donate_argnums=(0,))
        self._sample_indices_when_full = jax.jit(self._sample_indices_when_full)

    def store(self, state: np.ndarray, action: int, reward: float, done: bool,
              new_state: np.ndarray):
        entry = jnp.array((*state, action, reward, done, *new_state), dtype=jnp.float32)
        self._container, self._weights, self._index = self._store(self._container, self._weights, self._index, entry)
        self._full = self._max_capacity or self._index == self._max_capacity
        self._index %= self._max_capacity

    def _store(self, container: jax.Array, weights: jax.Array, index: jax.Array, entry: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        container = container.at[index].set(entry)
        weights = weights.at[index].set(jnp.where(index > 0, jnp.max(weights), 1.))
        index += 1
        return container, weights, index

    def batch_sample(self) -> jax.Array:
        if self._full:
            self._key, key = jax.random.split(self._key)
            self._indices = self._sample_indices_when_full(self._weights, key)
        elif not self._full and self.batch_size > self._index:
            raise ValueError("Not enough samples in the buffer")
        else:
            # with jax.disable_jit():
            #     self._key, key = jax.random.split(self._key)
            #     self._indices = jax.random.choice(
            #         key, 
            #         jnp.arange(self._index), 
            #         (self.batch_size,),
            #         replace=False,
            #         p=self._weights[:self._index] / jnp.sum(self._weights[:self._index])
            #     )
            self._key, key = jax.random.split(self._key)
            self._indices = self._sample_indices_when_full(self._weights, key)
        return self._container[self._indices]

    def _sample_indices_when_full(self, weights: jax.Array, key: jax.Array) -> jax.Array:
        return jax.random.choice(key, self._max_capacity, (self.batch_size,), replace=False, p=weights / jnp.sum(weights))

    def update_batch_weights(self, losses: jax.Array) -> None:
        if self.batch_size == len(losses):
            self._weights = self._jitted_update_batch_weights(self._weights, self._indices, losses)
        else:
            self._weights = self._update_batch_weights(self._weights, self._indices, losses)

    def _update_batch_weights(self, weights:jax.Array, indices: jax.Array, losses: jax.Array) -> jax.Array:
        batch_weights = jnp.abs(losses) + self.eps
        weights = weights.at[indices].set(batch_weights)
        return weights

    def __len__(self) -> int:
        return self._max_capacity if self._full else self._index.item()
