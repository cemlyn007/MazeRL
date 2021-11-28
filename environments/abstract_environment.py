import cv2
import numpy as np


class AbstractEnvironment:

    def __init__(self, display, magnification, window_name):
        self.window_name = window_name
        self.display = display
        self._magnification = magnification
        self.init_state = None
        self.goal_state = None
        self.width = 1.0
        self.height = 1.0
        self.image = np.zeros([int(self.magnification * self.height),
                               int(self.magnification * self.width), 3],
                              dtype=np.uint8)

    @property
    def magnification(self):
        return self._magnification

    @magnification.setter
    def magnification(self, value):
        self._magnification = value

    def draw_goal(self):
        raise NotImplementedError

    def draw_agent(self, agent_state):
        raise NotImplementedError

    def draw_environ(self):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        return self.init_state

    def draw(self, agent_state):
        raise NotImplementedError

    def show(self):
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)
