import numpy as np
import jax
from environments import abstract_environment, renderer, jax_random_environment


class RandomEnvironment(abstract_environment.AbstractEnvironment):
    def __init__(self, display: bool, magnification: int):
        self._stateless_environment = jax_random_environment.RandomEnvironment()
        self.display = display
        self.renderer = renderer.EnvironmentRenderer(
            "Random Environment",
            magnification,
            np.array(self._stateless_environment.free_blocks),
            np.array(self._stateless_environment.goal_state)
        )
        self._reset = jax.jit(self._stateless_environment.reset)
        self._step = jax.jit(self._stateless_environment.step)

    def reset(self) -> np.ndarray:
        return np.array(self._reset())

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        next_state, distance_to_goal = self._step(state, action)
        if self.display:
            self.renderer.draw(next_state)
            self.renderer.show()
        return np.array(next_state), distance_to_goal.item()

    def draw(self, state: np.ndarray) -> None:
        self.renderer.draw(state)
