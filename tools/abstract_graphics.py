import cv2
import numpy as np

import abstract_agent

Point = tuple[float, float]


class AbstractGraphics:

    def __init__(self, window_name: str, magnification: int,
                 agent: abstract_agent.AbstractAgent = None):
        self.magnification = magnification
        self.agent = agent
        self.image = None
        self.window_name = window_name

    def draw(self) -> None:
        raise NotImplementedError

    def show(self) -> None:
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        next_state, distance_from_goal = self.agent.environment.step(state, action)
        return next_state, distance_from_goal

    def save_image(self, filename: str) -> None:
        cv2.imwrite(filename, self.image)

    def convert(self, pt: Point) -> Point:
        mg = self.agent.environment.magnification
        return round(pt[0] * mg), round(mg * (1. - pt[1]))
