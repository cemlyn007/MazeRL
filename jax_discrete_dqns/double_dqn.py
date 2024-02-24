import torch
import helpers
import jax.numpy as jnp
import jax
from jax_discrete_dqns import dqn_with_target_network
import optax


class DiscreteDoubleDQN(dqn_with_target_network.DiscreteDQNWithTargetNetwork):
    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device):
        super().__init__(hps, n_actions, device)
        self._update = jax.jit(self._jax_update, donate_argnums=(0, 2))

    def compute_losses(self, params, target_params, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        q_values = self.q_network.apply(params, states)
        selected_q_values = jnp.take_along_axis(q_values, actions, axis=1)
        selected_q_values = jnp.squeeze(selected_q_values, -1)

        next_q_values = self.q_network.apply(params, next_states)
        target_next_q_values = self.q_network.apply(target_params, next_states)
        target_next_q_values = jax.lax.stop_gradient(target_next_q_values)

        selected_target_next_q_values_indices = jnp.argmax(next_q_values, axis=1, keepdims=True)

        selected_target_next_q_values = jnp.take_along_axis(
            target_next_q_values, selected_target_next_q_values_indices,
            axis=1
        )
        selected_target_next_q_values = jnp.squeeze(selected_target_next_q_values, -1)

        # TODO: Support terminal data as well!
        targets = rewards + self.hps.gamma * selected_target_next_q_values * (1 - dones)
        loss = jnp.abs(selected_q_values - targets)
        return loss.sum(), loss

    def _jax_update(self, params, target_params, optimizer_state, transition: jax.Array) -> tuple[any, any, jax.Array]:
        gradient, losses = jax.grad(self.compute_losses, has_aux=True)(params, target_params, transition)
        update, new_optimizer_state = self.optimizer.update(
            gradient, optimizer_state)
        new_params = optax.apply_updates(params, update)
        return new_params, new_optimizer_state, losses
