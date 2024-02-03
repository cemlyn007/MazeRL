import cv2
import numpy as np

from environments import abstract_environment, utils


class RandomEnvironment(abstract_environment.AbstractEnvironment):

    def __init__(self, display: bool, magnification: int):
        super().__init__(display, magnification, 'Random Environment')
        self.init_state, self.free_blocks, self.goal_state = utils.get_environment_space()

        self.agent_radius = int(0.01 * self.magnification)
        self.agent_colour = (0, 0, 255)
        self.goal_radius = int(0.01 * self.magnification)
        self.goal_colour = (0, 255, 0)

        self._predrawn_environ = np.zeros_like(self.image, dtype=np.uint8)
        self.predraw_environ()

    @abstract_environment.AbstractEnvironment.magnification.setter
    def magnification(self, value: int) -> None:
        self._magnification = value
        self.image = np.zeros([int(self.magnification * self.width),
                               int(self.magnification * self.height),
                               3], dtype=np.uint8)
        self._predrawn_environ = np.zeros_like(self.image)
        self.predraw_environ()

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
            self.draw(next_state)
            self.show()
        return next_state, distance_to_goal

    def predraw_environ(self) -> None:
        self._predrawn_environ.fill(100)
        for block in self.free_blocks:
            top_left = (int(self.magnification * block[0][0]),
                        int(self.magnification * (1 - block[0][1])))
            bottom_right = (int(self.magnification * block[1][0]),
                            int(self.magnification * (1 - block[1][1])))
            cv2.rectangle(self._predrawn_environ, top_left, bottom_right,
                          (0, 0, 0), thickness=cv2.FILLED)

    def draw_environ(self) -> None:
        self.image[:] = self._predrawn_environ

    def draw_agent(self, agent_state: np.ndarray) -> None:
        agent_centre = (int(agent_state[0] * self.magnification),
                        int((1 - agent_state[1]) * self.magnification))
        cv2.circle(self.image, agent_centre, self.agent_radius,
                   self.agent_colour, cv2.FILLED)

    def draw_goal(self) -> None:
        goal_centre = (int(self.goal_state[0] * self.magnification),
                       int((1 - self.goal_state[1]) * self.magnification))
        cv2.circle(self.image, goal_centre, self.goal_radius,
                   self.goal_colour, cv2.FILLED)

    def draw(self, agent_state: np.ndarray) -> None:
        self.draw_environ()
        self.draw_agent(agent_state)
        self.draw_goal()
