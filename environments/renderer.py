import cv2
import numpy as np


class EnvironmentRenderer:
    def __init__(
        self,
        window_name: str,
        magnification: int,
        free_blocks: np.ndarray,
        goal_state: np.ndarray,
    ) -> None:
        self._window_name = window_name
        self._magnification = magnification
        self._free_blocks = free_blocks
        self._goal_state = goal_state
        self.agent_radius = int(0.01 * self._magnification)
        self.agent_colour = (0, 0, 255)
        self.goal_radius = int(0.01 * self._magnification)
        self.goal_colour = (0, 255, 0)
        self.image = np.zeros(
            [
                int(self._magnification),
                int(self._magnification),
                3,
            ],
            dtype=np.uint8,
        )
        self._predrawn_environ = np.zeros_like(self.image, dtype=np.uint8)
        self._predraw_environ()

    def draw(self, agent_state: np.ndarray) -> None:
        self._draw_environ()
        self._draw_agent(agent_state)
        self._draw_goal()

    def show(self) -> None:
        cv2.imshow(self._window_name, self.image)
        cv2.waitKey(1)

    def _predraw_environ(self) -> None:
        self._predrawn_environ.fill(100)
        for block in self._free_blocks:
            top_left = (
                int(self._magnification * block[0][0]),
                int(self._magnification * (1 - block[0][1])),
            )
            bottom_right = (
                int(self._magnification * block[1][0]),
                int(self._magnification * (1 - block[1][1])),
            )
            cv2.rectangle(
                self._predrawn_environ,
                top_left,
                bottom_right,
                (0, 0, 0),
                thickness=cv2.FILLED,
            )
        self.image[:] = self._predrawn_environ
        self._draw_goal()

    def _draw_environ(self) -> None:
        np.copyto(self.image, self._predrawn_environ)

    def _draw_agent(self, agent_state: np.ndarray) -> None:
        agent_centre = (
            int(agent_state[0] * self._magnification),
            int((1 - agent_state[1]) * self._magnification),
        )
        cv2.circle(
            self.image, agent_centre, self.agent_radius, self.agent_colour, cv2.FILLED
        )

    def _draw_goal(self) -> None:
        goal_centre = (
            int(self._goal_state[0] * self._magnification),
            int((1 - self._goal_state[1]) * self._magnification),
        )
        cv2.circle(
            self.image, goal_centre, self.goal_radius, self.goal_colour, cv2.FILLED
        )
