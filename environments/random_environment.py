import numpy as np

from environments import abstract_environment, utils, renderer


class RandomEnvironment(abstract_environment.AbstractEnvironment):

    def __init__(self, display: bool, magnification: int):
        self.display = display
        self.init_state, self.free_blocks, self.goal_state = (
            utils.get_environment_space()
        )
        self.renderer = renderer.EnvironmentRenderer(
            "Random Environment", magnification, self.free_blocks, self.goal_state
        )

    def reset(self) -> np.ndarray:
        return self.init_state

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        next_state = np.where(np.linalg.norm(action) > 0.02, state, state + action)
        in_bounds = np.any(
            np.logical_and(
                np.logical_and(
                    self.free_blocks[:, 0, 0] < next_state[0],
                    next_state[0] < self.free_blocks[:, 1, 0],
                ),
                np.logical_and(
                    self.free_blocks[:, 1, 1] < next_state[1],
                    next_state[1] < self.free_blocks[:, 0, 1],
                ),
            )
        )
        next_state = np.where(in_bounds, next_state, state)
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        if self.display:
            self.renderer.draw(next_state)
            self.renderer.show()
        return next_state, distance_to_goal

    def draw(self, state: np.ndarray) -> None:
        self.renderer.draw(state)