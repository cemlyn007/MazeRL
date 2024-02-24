import torch
import optax
import helpers
from abstract_dqns import dqn
from jax_discrete_dqns import network
import jax
import jax.numpy as jnp


class DiscreteDQN(dqn.AbstractDQN):
    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device):
        self.device = device
        self.hps = hps
        self.q_network = network.DiscreteNetwork(2, n_actions)
        self._params = self.q_network.init(
            jax.random.PRNGKey(0), jnp.ones((1, 2), jnp.float32))
        self.optimizer = optax.sgd(self.hps.lr)
        self.optimizer_state = self.optimizer.init(self._params)
        self._device = jax.devices('cpu' if device == torch.device('cpu') else 'gpu')[0]
        self._update = jax.jit(self._jax_update, donate_argnums=(0, 1))

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def has_target_network(self) -> bool:
        return False

    def train_q_network(self, transition: torch.Tensor) -> torch.Tensor:
        jax_transition = jnp.array(transition.numpy(), jnp.float32)
        jax_transition = jax.device_put(jax_transition, self._device)
        self._params, self.optimizer_state, losses = self._update(
            self._params, self.optimizer_state, jax_transition)
        return torch.tensor(losses)
    
    def _jax_update(self, params, optimizer_state, transition: jax.Array) -> tuple[any, any, jax.Array]:
        gradient, losses = jax.grad(self.compute_losses, has_aux=True)(params, transition)
        update, new_optimizer_state = self.optimizer.update(
            gradient, params, optimizer_state)
        new_params = optax.apply_updates(params, update)
        return new_params, new_optimizer_state, losses

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
