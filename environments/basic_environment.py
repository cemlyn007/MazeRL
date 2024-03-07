import numpy as np

from environments import abstract_environment, renderer


class BasicEnvironment(abstract_environment.AbstractEnvironment):
    def __init__(self, display: bool, magnification: int):
        self.display = display
        self._init_state = np.array([0.15, 0.15], dtype=np.float32)
        self._goal_state = np.array([0.75, 0.85], dtype=np.float32)
        self._obstacle_space = np.array([[0.3, 0.5], [0.3, 0.6]], dtype=np.float32)
        self.renderer = renderer.EnvironmentRenderer(
            "Basic Environment",
            magnification,
            np.expand_dims(self._obstacle_space, 0),
            self._goal_state,
        )

    def reset(self) -> np.ndarray:
        if self.display:
            self.renderer.draw(self._init_state)
            self.renderer.show()
        return self._init_state

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        next_state = state + action
        if (
            next_state[0] < 0.0
            or next_state[0] > 1.0
            or next_state[1] < 0.0
            or next_state[1] > 1.0
        ):
            next_state = state
        if (
            self._obstacle_space[0, 0] <= next_state[0] < self._obstacle_space[0, 1]
        ) and (
            self._obstacle_space[1, 0] <= next_state[1] < self._obstacle_space[1, 1]
        ):
            next_state = state
        distance_to_goal = np.linalg.norm(next_state - self._goal_state)
        if self.display:
            self.renderer.draw(next_state)
            self.renderer.show()
        return next_state, distance_to_goal.item()
