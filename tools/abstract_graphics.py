import cv2
import numpy as np
import copy


class AbstractGraphics:

    def __init__(self, window_name, magnification, agent=None):
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

    def draw(self):
        raise NotImplementedError

    def show(self):
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)

    def step(self, state, discrete_action):
        cont_action = self.agent._discrete_action_to_continuous(discrete_action)
        next_state, distance_from_goal = self.environment.step(state,
                                                               cont_action)
        return next_state, distance_from_goal

    def save_image(self, filename):
        cv2.imwrite(filename, self.image)
