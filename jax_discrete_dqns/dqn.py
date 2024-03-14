import optax
import helpers
from abstract_dqns import dqn
from jax_discrete_dqns import network
import jax
import jax.numpy as jnp
from typing import TypeVar
import abc
from flax.core import scope


T = TypeVar('T')


class StatelessDiscreteDQN:
    @property
    def has_target_network(self) -> bool:
        return False

    @abc.abstractmethod
    def predict_q_values(self, network: scope.VariableDict, observations: jax.Array) -> jax.Array:
        pass

    @abc.abstractmethod
    def train_q_network(self, state: T, transition: jax.Array) -> tuple[T, jax.Array]:
        pass


class DiscreteDQN(dqn.AbstractDQN):
    def __init__(self, hps: helpers.JaxHyperparameters, n_actions: int, device: jax.Device):
        self.hps = hps
        self.q_network = network.DiscreteNetwork(2, n_actions)
        self._params = self.q_network.init(jax.random.PRNGKey(0), jnp.ones((1, 2), jnp.float32))
        self.optimizer = optax.sgd(self.hps.lr)
        self.optimizer_state = self.optimizer.init(self._params)
        self._device = device
        self._update = jax.jit(self._jax_update, donate_argnums=(0, 1))

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def has_target_network(self) -> bool:
        return False

    def predict_q_values(self, observations: jax.Array) -> jax.Array:
        return self.q_network.apply(self._params, observations)

    def train_q_network(self, transition: jax.Array) -> jax.Array:
        jax_transition = jax.device_put(transition, self._device)
        self._params, self.optimizer_state, losses = self._update(self._params, self.optimizer_state, jax_transition)
        return losses

    def _jax_update(self, params, optimizer_state, transition: jax.Array) -> tuple[any, any, jax.Array]:
        batch_size = transition.shape[0]
        transition = jax.tree_map(lambda x: jnp.reshape(x, (self.hps.mini_batches, -1, *x.shape[1:])), transition)

        def body_fun(i, carry):
            params, optimizer_state, transition, losses = carry
            mini_batch_transition = jax.tree_map(lambda x: x[i], transition)
            gradient, mini_batch_losses = jax.grad(self.compute_losses, has_aux=True)(params, mini_batch_transition)
            update, new_optimizer_state = self.optimizer.update(gradient, params, optimizer_state)
            new_params = optax.apply_updates(params, update)
            losses = losses.at[i].set(mini_batch_losses)
            return new_params, new_optimizer_state, losses

        new_params, new_optimizer_state, losses = jax.lax.fori_loop(
            0,
            self.hps.mini_batches,
            body_fun,
            (params, optimizer_state, transition, jnp.empty(transition[0].shape[0], jnp.float32))
        )
        return new_params, new_optimizer_state, jnp.reshape(losses, (batch_size,))

    def compute_losses(self, params, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        q_values = self.q_network.apply(params, states)
        selected_q_values = jnp.take_along_axis(q_values, actions, axis=1).flatten()
        next_q_values = self.q_network.apply(params, next_states)
        max_q_values = jnp.max(next_q_values, axis=1)
        targets = rewards + self.hps.gamma * max_q_values * (1 - dones)
        loss = jnp.abs(selected_q_values - targets)
        return loss.sum(), loss

    @staticmethod
    def _unpack_transitions(transitions: jax.Array) -> tuple[jax.Array, jax.Array,
                                                             jax.Array,
                                                             jax.Array, jax.Array]:
        states = transitions[:, :2]
        actions = transitions[:, 2].astype(jnp.int32)
        actions = jnp.expand_dims(actions, axis=-1)
        rewards = transitions[:, 3]
        dones = transitions[:, 4]
        next_states = transitions[:, 5:]
        return states, actions, rewards, dones, next_states
