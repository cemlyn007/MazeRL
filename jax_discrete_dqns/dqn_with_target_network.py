import copy

import torch
import jax
import optax
import helpers
from jax_discrete_dqns import dqn
import jax.numpy as jnp
import numpy as np

class DiscreteDQNWithTargetNetwork(dqn.DiscreteDQN):
    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device):
        super().__init__(hps, n_actions, device)
        self._target_params = copy.deepcopy(self._params)
        self._update = jax.jit(self._jax_update, donate_argnums=(0, 2))

    @property
    def has_target_network(self) -> bool:
        return True
    
    def train_q_network(self, transition: torch.Tensor) -> torch.Tensor:
        jax_transition = jnp.array(transition.numpy(), jnp.float32)
        jax_transition = jax.device_put(jax_transition, self._device)
        self._params, self.optimizer_state, losses = self._update(
            self._params, self._target_params, self.optimizer_state, jax_transition)
        return torch.tensor(np.array(losses))

    def compute_losses(self, params, target_params, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        states, actions, rewards, dones, next_states = self._unpack_transitions(
            transitions)
        q_values = self.q_network.apply(params, states)
        selected_q_values = jnp.take_along_axis(q_values, actions, axis=1).flatten()
        next_q_values = self.q_network.apply(target_params, next_states)
        next_q_values = jax.lax.stop_gradient(next_q_values)
        max_q_values = jnp.max(next_q_values, axis=1)
        targets = rewards + self.hps.gamma * max_q_values * (1 - dones)
        loss = jnp.abs(selected_q_values - targets)
        return loss.sum(), loss
    
    def _jax_update(self, params, target_params, optimizer_state, transition: jax.Array) -> tuple[any, any, jax.Array]:
        gradient, losses = jax.grad(self.compute_losses, has_aux=True)(params, target_params, transition)
        update, new_optimizer_state = self.optimizer.update(
            gradient, params, optimizer_state)
        new_params = optax.apply_updates(params, update)
        return new_params, new_optimizer_state, losses

    def update_target_network(self) -> None:
        self._target_params = copy.deepcopy(self._params)
