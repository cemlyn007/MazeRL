import copy

import jax
import optax
import helpers
from jax_discrete_dqns import dqn
import jax.numpy as jnp


class DiscreteDQNWithTargetNetwork(dqn.DiscreteDQN):
    def __init__(self, hps: helpers.JaxHyperparameters, n_actions: int, device: jax.Device):
        super().__init__(hps, n_actions, device)
        self._target_params = copy.deepcopy(self._params)
        self._update = jax.jit(self._jax_update, donate_argnums=(0, 2))

    @property
    def has_target_network(self) -> bool:
        return True

    def train_q_network(self, transition: jax.Array) -> jax.Array:
        jax_transition = jax.device_put(transition, self._device)
        self._params, self._target_params, self.optimizer_state, losses = self._update(
            self._params, self._target_params, self.optimizer_state, jax_transition
        )
        return losses

    def compute_losses(self, params, target_params, transitions: jax.Array) -> tuple[jax.Array, jax.Array]:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        q_values = self.q_network.apply(params, states)
        selected_q_values = jnp.take_along_axis(q_values, actions, axis=1).flatten()
        next_q_values = self.q_network.apply(target_params, next_states)
        next_q_values = jax.lax.stop_gradient(next_q_values)
        max_q_values = jnp.max(next_q_values, axis=1)
        targets = rewards + self.hps.gamma * max_q_values * (1 - dones)
        loss = jnp.abs(selected_q_values - targets)
        return loss.sum(), loss
    
    def _jax_update(self, params, target_network, optimizer_state, transition: jax.Array) -> tuple[any, any, any, jax.Array]:
        batch_size = transition.shape[0]
        transition = jax.tree_map(lambda x: jnp.reshape(x, (self.hps.mini_batches, -1, *x.shape[1:])), transition)

        def body_fun(i, carry):
            params, target_network, optimizer_state, transition, losses = carry
            mini_batch_transition = jax.tree_map(lambda x: x[i], transition)
            gradient, mini_batch_losses = jax.grad(self.compute_losses, has_aux=True)(params, target_network, mini_batch_transition)
            update, new_optimizer_state = self.optimizer.update(gradient, params, optimizer_state)
            new_params = optax.apply_updates(params, update)
            losses = losses.at[i].set(mini_batch_losses)
            return new_params, target_network, new_optimizer_state, losses

        new_params, target_network, new_optimizer_state, losses = jax.lax.fori_loop(
            0,
            self.hps.mini_batches,
            body_fun,
            (params, target_network, optimizer_state, transition, jnp.empty(transition[0].shape[:2], jnp.float32))
        )
        return new_params, target_network, new_optimizer_state, jnp.reshape(losses, (batch_size,))

    def update_target_network(self) -> None:
        self._target_params = copy.deepcopy(self._params)
