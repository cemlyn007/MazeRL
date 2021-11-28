import cv2
import numpy as np

from environments import abstract_environment


class BasicEnvironment(abstract_environment.AbstractEnvironment):

    def __init__(self, display: bool, magnification: int):
        super().__init__(display, magnification, "Basic Environment")
        self.init_state = np.array([0.15, 0.15], dtype=np.float32)
        self.goal_state = np.array([0.75, 0.85], dtype=np.float32)
        self.obstacle_space = np.array([[0.3, 0.5], [0.3, 0.6]],
                                       dtype=np.float32)

        self.goal_colour = (0, 255, 0)
        self.agent_colour = (0, 0, 255)
        self.environ_image = None
        self.update_environ_image()

    @abstract_environment.AbstractEnvironment.magnification.setter
    def magnification(self, value):
        self._magnification = value
        self.update_environ_image()

    def step(self, state, action):
        next_state = state + action
        if (next_state[0] < 0.0 or next_state[0] > 1.0
                or next_state[1] < 0.0 or next_state[1] > 1.0):
            next_state = state
        if ((self.obstacle_space[0, 0] <= next_state[0]
             < self.obstacle_space[0, 1]) and (self.obstacle_space[1, 0]
                                               <= next_state[1]
                                               < self.obstacle_space[1, 1])):
            next_state = state
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        if self.display:
            self.draw(next_state)
            self.show()
        return next_state, distance_to_goal

    def update_environ_image(self):
        self.environ_image = np.zeros([int(self.magnification * self.height),
                                       int(self.magnification * self.width), 3],
                                      dtype=np.uint8)
        self.environ_image.fill(0)
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification *
                             (self.obstacle_space[0, 1]
                              - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification *
                              (self.obstacle_space[1, 1]
                               - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width,
                                 obstacle_top + obstacle_height)
        cv2.rectangle(self.environ_image, obstacle_top_left, obstacle_bottom_right,
                      (150, 150, 150), thickness=cv2.FILLED)

    def draw_environ(self):
        self.image = self.environ_image.copy()

    def draw_agent(self, agent_state):
        agent_centre = (int(agent_state[0] * self.magnification),
                        int((1 - agent_state[1]) * self.magnification))
        agent_radius = int(0.02 * self.magnification)
        cv2.circle(self.image, agent_centre, agent_radius, self.agent_colour,
                   cv2.FILLED)

    def draw_goal(self):
        goal_centre = (int(self.goal_state[0] * self.magnification),
                       int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.02 * self.magnification)

        cv2.circle(self.image, goal_centre, goal_radius, self.goal_colour,
                   cv2.FILLED)

    def draw(self, agent_state):
        self.draw_environ()
        self.draw_agent(agent_state)
        self.draw_goal()
