import cv2
import numpy as np

from environments import abstract_environment


class RandomEnvironment(abstract_environment.AbstractEnvironment):

    def __init__(self, display: bool, magnification: int):
        super().__init__(display, magnification, 'Random Environment')
        self.init_state, self.free_blocks, self.goal_state = self._get_environment_space()

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

    def _get_environment_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        init_state_x = 0.05
        init_state_y = np.random.uniform(0.05, 0.95)
        init_state = np.array([init_state_x, init_state_y],
                                   dtype=np.float32)
        free_blocks = []
        block_bottom = init_state_y - np.random.uniform(0.1, 0.2)
        block_top = init_state_y + np.random.uniform(0.1, 0.2)
        block_left = 0.02
        block_right = block_left + np.random.uniform(0.03, 0.1)
        top_left = (block_left, block_top)
        bottom_right = (block_right, block_bottom)
        block = (top_left, bottom_right)
        free_blocks.append(block)
        prev_top = top_left[1]
        prev_bottom = bottom_right[1]
        prev_right = bottom_right[0]
        while prev_right < 0.8:
            is_within_boundary = False
            while not is_within_boundary:
                block_height = np.random.uniform(0.05, 0.4)
                block_bottom_max = prev_top - 0.05
                block_bottom_min = prev_bottom - (block_height - 0.05)
                block_bottom_mid = 0.5 * (block_bottom_min + block_bottom_max)
                block_bottom_half_range = block_bottom_max - block_bottom_mid
                r1 = np.random.uniform(-block_bottom_half_range,
                                       block_bottom_half_range)
                r2 = np.random.uniform(-block_bottom_half_range,
                                       block_bottom_half_range)
                if np.fabs(r1) > np.fabs(r2):
                    block_bottom = block_bottom_mid + r1
                else:
                    block_bottom = block_bottom_mid + r2
                block_top = block_bottom + block_height
                block_left = prev_right
                block_width = np.random.uniform(0.03, 0.1)
                block_right = block_left + block_width
                top_left = (block_left, block_top)
                bottom_right = (block_right, block_bottom)
                if (block_bottom < 0 or block_top > 1 or
                        block_left < 0 or block_right > 1):
                    is_within_boundary = False
                else:
                    is_within_boundary = True
            block = (top_left, bottom_right)
            free_blocks.append(block)
            prev_top = block_top
            prev_bottom = block_bottom
            prev_right = block_right
        block_height = np.random.uniform(0.05, 0.15)
        block_bottom_max = prev_top - 0.05
        block_bottom_min = prev_bottom - (block_height - 0.05)
        block_bottom = np.random.uniform(block_bottom_min, block_bottom_max)
        block_top = block_bottom + block_height
        block_left = prev_right
        block_right = 0.98
        top_left = (block_left, block_top)
        bottom_right = (block_right, block_bottom)
        block = (top_left, bottom_right)
        free_blocks.append(block)
        free_blocks = np.array(free_blocks, dtype=np.float32)
        y_goal_state = np.random.uniform(block_bottom + 0.01, block_top - 0.01)
        goal_state = np.array([0.95, y_goal_state], dtype=np.float32)
        return init_state, free_blocks, goal_state

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
