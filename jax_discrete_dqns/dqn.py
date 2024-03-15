import optax
import helpers
from abstract_dqns import dqn
from jax_discrete_dqns import network
import jax
import jax.numpy as jnp
from typing import TypeVar, NamedTuple
from flax.core import scope

T = TypeVar('T')


class State(NamedTuple):
    network: scope.VariableDict
    optimizer: optax.OptState


class StatelessDiscreteDQN:
    def __init__(self, hps: helpers.JaxHyperparameters, n_actions: int) -> None:
        self.hps = hps
        self.q_network = network.DiscreteNetwork(2, n_actions)
        self.optimizer = optax.sgd(self.hps.lr)

    @property
    def has_target_network(self) -> bool:
        return False
    
    def reset(self) -> State:
        params = self.q_network.init(jax.random.PRNGKey(0), jnp.ones((1, 2), jnp.float32))
        optimizer_state = self.optimizer.init(params)
        return State(network=params, optimizer=optimizer_state)

    def predict_q_values(self, network: scope.VariableDict, observations: jax.Array) -> jax.Array:
        q_values: jax.Array = self.q_network.apply(network, observations)
        return q_values

    def train_q_network(self, state: State, transition: jax.Array) -> tuple[T, jax.Array]:
        batch_size = transition.shape[0]
        transition = jax.tree_map(lambda x: jnp.reshape(x, (self.hps.mini_batches, -1, *x.shape[1:])), transition)

        def body_fun(i, carry: tuple[State, jax.Array, jax.Array]) -> tuple[State, jax.Array, jax.Array]:
            state, transition, losses = carry
            mini_batch_transition = jax.tree_map(lambda x: x[i], transition)
            gradient, mini_batch_losses = jax.grad(self.compute_losses, has_aux=True)(state.network, mini_batch_transition)
            update, new_optimizer_state = self.optimizer.update(gradient, state.network, state.optimizer)
            new_params = optax.apply_updates(state.network, update)
            losses = losses.at[i].set(mini_batch_losses)
            new_state = State(network=new_params, optimizer=new_optimizer_state)
            return new_state, transition, losses

        new_state, _, losses = jax.lax.fori_loop(
            0,
            self.hps.mini_batches,
            body_fun,
            (state, transition, jnp.empty(transition[0].shape[0], jnp.float32))
        )
        return new_state, jnp.reshape(losses, (batch_size,))
    
    def compute_losses(self, network: scope.VariableDict, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        q_values = self.q_network.apply(network, states)
        selected_q_values = jnp.take_along_axis(q_values, actions, axis=1).flatten()
        next_q_values = self.q_network.apply(network, next_states)
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


class DiscreteDQN(dqn.AbstractDQN):
    def __init__(self, hps: helpers.JaxHyperparameters, n_actions: int, device: jax.Device):
        self.hps = hps
        self._stateless_dqn = StatelessDiscreteDQN(hps, n_actions)
        self._state = self._stateless_dqn.reset()
        self._device = device
        self._update = jax.jit(self._stateless_dqn.train_q_network, donate_argnums=(0,))
        self._predict_q_values = jax.jit(self._stateless_dqn.predict_q_values)

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def has_target_network(self) -> bool:
        return self._stateless_dqn.has_target_network

    def predict_q_values(self, observations: jax.Array) -> jax.Array:
        return self._predict_q_values(self._state, observations)

    def train_q_network(self, transition: jax.Array) -> jax.Array:
        jax_transition = jax.device_put(transition, self._device)
        self._state, losses = self._update(self._state, jax_transition)
        return losses

    def compute_losses(self, params, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self._stateless_dqn.compute_losses(params, transitions)
