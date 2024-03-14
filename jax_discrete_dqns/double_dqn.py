from typing import NamedTuple

from torch import Tensor
import helpers
import jax.numpy as jnp
import jax
import abstract_dqns.dqn
from jax_discrete_dqns import network, dqn
import optax
import helpers
from flax.core import scope

class State(NamedTuple):
    network: scope.VariableDict
    target_network: scope.VariableDict
    optimizer: optax.OptState


class StatelessDiscreteDoubleDQN(dqn.StatelessDiscreteDQN):
    def __init__(self, hps: helpers.JaxHyperparameters, n_actions: int):
        self._hps = hps
        self._n_actions = n_actions
        self.q_network = network.DiscreteNetwork(2, n_actions)
        self._params = self.q_network.init(jax.random.PRNGKey(0), jnp.ones((1, 2), jnp.float32))
        self.optimizer = optax.sgd(self._hps.lr)

    def reset(self) -> State:
        params = self.q_network.init(jax.random.PRNGKey(0), jnp.ones((1, 2), jnp.float32))
        target_params = self.q_network.init(jax.random.PRNGKey(0), jnp.ones((1, 2), jnp.float32))
        optimizer_state = self.optimizer.init(self._params)
        return State(network=params, target_network=target_params, optimizer=optimizer_state)

    @property
    def has_target_network(self) -> bool:
        return True

    def predict_q_values(self, network: scope.VariableDict, observations: jax.Array) -> jax.Array:
        return self.q_network.apply(network, observations)

    def train_q_network(self, state: State, transition: jax.Array) -> tuple[State, jax.Array]:
        new_state, losses = self._jax_update(state, transition)
        return new_state, losses

    def compute_losses(self, network: scope.VariableDict, target_network: scope.VariableDict, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        q_values = self.q_network.apply(network, states)
        selected_q_values = jnp.take_along_axis(q_values, actions, axis=1)
        selected_q_values = jnp.squeeze(selected_q_values, -1)

        next_q_values = self.q_network.apply(network, next_states)
        target_next_q_values = self.q_network.apply(target_network, next_states)
        target_next_q_values = jax.lax.stop_gradient(target_next_q_values)

        selected_target_next_q_values_indices = jnp.argmax(next_q_values, axis=1, keepdims=True)

        selected_target_next_q_values = jnp.take_along_axis(
            target_next_q_values, selected_target_next_q_values_indices,
            axis=1
        )
        selected_target_next_q_values = jnp.squeeze(selected_target_next_q_values, -1)

        targets = rewards + self._hps.gamma * selected_target_next_q_values * (1 - dones)
        loss = jnp.abs(selected_q_values - targets)
        return loss.sum(), loss

    def _jax_update(self, state: State, transition: jax.Array) -> tuple[State, jax.Array]:
        batch_size = transition.shape[0]
        mini_batch_size, remainder = divmod(batch_size, self._hps.mini_batches)
        if remainder != 0:
            raise ValueError(f"Batch size {batch_size} is not divisible by mini-batch size {self._hps.mini_batches}")
        transition = jax.tree_map(lambda x: jnp.reshape(
            x, (self._hps.mini_batches, mini_batch_size, *x.shape[1:])), transition)

        def body_fun(i, carry: tuple[State, jax.Array, jax.Array]) -> tuple[State, jax.Array, jax.Array]:
            state, transition, losses = carry
            mini_batch_transition = jax.tree_map(lambda x: x[i], transition)
            gradient, mini_batch_losses = jax.grad(self.compute_losses, has_aux=True)(
                state.network, state.target_network, mini_batch_transition)
            update, new_optimizer_state = self.optimizer.update(gradient, state.optimizer)
            updated_network = optax.apply_updates(state.network, update)
            losses = losses.at[i].set(mini_batch_losses)
            new_state = State(network=updated_network, target_network=state.target_network,
                              optimizer=new_optimizer_state)
            return new_state, transition, losses

        new_state, _, losses = jax.lax.fori_loop(
            0,
            self._hps.mini_batches,
            body_fun,
            (state, transition, jnp.empty((transition.shape[:2]), jnp.float32))
        )
        return new_state, jnp.reshape(losses, (batch_size,))

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


class DiscreteDoubleDQN(abstract_dqns.dqn.AbstractDQN):
    def __init__(self, hps: helpers.JaxHyperparameters, n_actions: int, device: jax.Device):
        self._stateless_dqn = StatelessDiscreteDoubleDQN(hps, n_actions)
        self._state = self._stateless_dqn.reset()
        self._device = device
        self._train_q_network = jax.jit(self._stateless_dqn.train_q_network)
        self._compute_losses = jax.jit(self._stateless_dqn.compute_losses)
        self._predict = jax.jit(self._stateless_dqn.predict_q_values)

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def has_target_network(self) -> bool:
        return True
    
    def predict_q_values(self, observations: jax.Array) -> jax.Array:
        return self._predict(self._state.network, observations)

    def train_q_network(self, transition: jax.Array) -> jax.Array:
        jax_transition = jax.device_put(transition, self._device)
        self._state, losses = self._train_q_network(
            self._state,
            jax_transition,
        )
        return losses
    
    def compute_losses(self, transitions: Tensor) -> Tensor:
        return self._compute_losses(self._state.network, self._state.target_network, transitions)

    def update_target_network(self) -> None:
        self._state = self._state._replace(target_network=self._state.network)
