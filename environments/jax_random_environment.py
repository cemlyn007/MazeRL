import jax.numpy as jnp
import jax

from environments import utils


class RandomEnvironment:
    def __init__(self):
        init_state, free_blocks, goal_state = (
            utils.get_environment_space()
        )
        self.init_state = jnp.array(init_state)
        self.free_blocks = jnp.array(free_blocks)
        self.goal_state = jnp.array(goal_state)

    def reset(self) -> jax.Array:
        return self.init_state

    def step(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_state = jnp.where(jnp.linalg.norm(action) > 0.02, state, state + action)
        in_bounds = jnp.any(
            jnp.logical_and(
                jnp.logical_and(
                    self.free_blocks[:, 0, 0] < next_state[0],
                    next_state[0] < self.free_blocks[:, 1, 0],
                ),
                jnp.logical_and(
                    self.free_blocks[:, 1, 1] < next_state[1],
                    next_state[1] < self.free_blocks[:, 0, 1],
                ),
            )
        )
        next_state = jnp.where(in_bounds, next_state, state)
        distance_to_goal = jnp.linalg.norm(next_state - self.goal_state)
        return next_state, distance_to_goal
