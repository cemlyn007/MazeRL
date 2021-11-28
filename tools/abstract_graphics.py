import copy

import cv2
import numpy as np

import abstract_agent


class AbstractGraphics:

    def __init__(self, window_name: str, magnification: int,
                 agent: abstract_agent.AbstractAgent = None):
        self.magnification = magnification
        self.width = self.height = 1.0
        self.environment = copy.deepcopy(agent.environment)
        self.environment.display = False
        self.environment.magnification = self.magnification
        self.environment.image = np.zeros([int(magnification * self.width),
                                           int(magnification * self.height),
                                           3], dtype=np.uint8)
        self.agent = agent
        self.dqn = agent.dqn
        self.image = None
        self.window_name = window_name

    def draw(self) -> None:
        raise NotImplementedError

    def show(self) -> None:
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        next_state, distance_from_goal = self.environment.step(state, action)
        return next_state, distance_from_goal

    def save_image(self, filename: str) -> None:
        cv2.imwrite(filename, self.image)
