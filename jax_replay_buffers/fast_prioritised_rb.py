from typing import NamedTuple
import numpy as np
import jax

from jax_replay_buffers import abstract_replay_buffer
import functools
import operator
import jax.numpy as jnp


class State(NamedTuple):
    index: jax.Array
    buffer: jax.Array
    weights: jax.Array


class StatelessFastPrioritisedExperienceReplayBuffer:
    def __init__(self, max_capacity: int, batch_size: int, eps: float, state_shape: tuple[int, ...]):
        self._batch_size = batch_size
        self._max_capacity = max_capacity
        self._full = False
        self._state_size = functools.reduce(operator.mul, state_shape, 1)
        self._container_shape = (max_capacity, self._state_size + 1 + 1 + 1 + self._state_size)
        self.eps = eps

    def reset(self) -> State:
        return State(
            index=jnp.array(0, dtype=jnp.uint32),
            buffer=jnp.empty(self._container_shape, dtype=jnp.float32),
            weights=jnp.zeros(self._max_capacity, dtype=jnp.float32),
        )

    def store(self, state: State, observation: jax.Array, action: jax.Array, reward: jax.Array, done: jax.Array,
              next_observation: jax.Array) -> State:
        if observation.dtype != jnp.float32:
            raise ValueError("Observation must be of type jnp.float32")
        if next_observation.dtype != jnp.float32:
            raise ValueError("Next observation must be of type jnp.float32")
        entry = jnp.array((*observation, action, reward, done, *next_observation), dtype=jnp.float32)
        if entry.shape != self._container_shape[1:]:
            raise ValueError(
                f"Entry shape must be {self._container_shape[1:]} but got {entry.shape}"
            )
        # else...
        buffer = state.buffer.at[state.index].set(entry)
        weights = state.weights.at[state.index].set(jnp.where(state.index > 0, jnp.max(state.weights), 1.0))
        index = (state.index + 1) % self._max_capacity
        return State(
            index,
            buffer,
            weights,
        )

    def batch_sample(self, state: State, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        probabilties = state.weights / jnp.sum(state.weights)
        indices = jax.random.choice(key, self._max_capacity, (self._batch_size,), replace=False, p=probabilties)
        return state.buffer[indices], indices

    def update_batch_weights(self, state: State, indices: jax.Array, losses: jax.Array) -> State:
        batch_weights = jnp.abs(losses) + self.eps
        weights = state.weights.at[indices].set(batch_weights)
        return state._replace(weights=weights)


class FastPrioritisedExperienceReplayBuffer(abstract_replay_buffer.ReplayBuffer):
    def __init__(self, max_capacity: int, batch_size: int, eps: float, state_shape: tuple[int, ...]):
        self._key = jax.random.PRNGKey(0)
        self._stateless = StatelessFastPrioritisedExperienceReplayBuffer(max_capacity, batch_size, eps, state_shape)

        self._stateless_state = self._stateless.reset()
        self._stateless_state_indices = None

        self._stateless_store = jax.jit(self._stateless.store)
        self._stateless_batch_sample = jax.jit(self._stateless.batch_sample)
        self._stateless_update_batch_weights = jax.jit(self._stateless.update_batch_weights)

    def store(self, state: np.ndarray, action: int, reward: float, done: bool, new_state: np.ndarray) -> None:
        self._stateless_state = self._stateless_store(
            self._stateless_state,
            jnp.array(state),
            jnp.array(action),
            jnp.array(reward),
            jnp.array(done),
            jnp.array(new_state)
        )

    def batch_sample(self) -> jax.Array:
        self._key, key = jax.random.split(self._key)
        batch, self._stateless_state_indices = self._stateless_batch_sample(self._stateless_state, key)
        return batch

    def update_batch_weights(self, losses: jax.Array) -> None:
        if self._stateless_state_indices is None:
            raise ValueError("Cannot update batch weights before sampling a batch")
        # else...
        self._stateless_state = self._stateless_update_batch_weights(
            self._stateless_state,
            self._stateless_state_indices,
            losses
        )
