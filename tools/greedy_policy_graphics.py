import cv2
import numpy as np
import torch

from tools.abstract_graphics import AbstractGraphics


class GreedyPolicyTool(AbstractGraphics):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    def __init__(self, magnification, agent, max_step_num=20):
        name = "Greedy Policy"
        super(GreedyPolicyTool, self).__init__(name, magnification, agent)
        self.image = self.environment.image
        self.max_step_num = max_step_num
        self.orb_radius = int(0.02 * self.magnification)

    def draw_blank_environment(self):
        self.environment.draw_environ()
        self.environment.draw_goal()
        self.image = self.environment.image

    def draw(self):
        self.draw_blank_environment()
        self.draw_greedy_path()

    def draw_greedy_path(self):
        red = np.flip(np.linspace(0, 255, num=self.max_step_num + 1,
                                  endpoint=True, dtype=np.uint8)).tolist()
        green = np.linspace(0, 255, num=self.max_step_num + 1, endpoint=True,
                            dtype=np.uint8).tolist()
        thickness = int(0.004 * self.magnification)

        state = self.agent.environment.init_state
        displayed_state = state * self.magnification
        displayed_state[1] = (self.height * self.magnification
                              - displayed_state[1])
        displayed_state = tuple(displayed_state.astype(int))
        policy_path = [displayed_state]

        was_training = False
        if self.dqn.q_network.training:
            self.dqn.q_network.eval()
            was_training = True

        for i in range(self.max_step_num + 1):
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(self.dqn.device)
                action = self.agent.get_greedy_action(state_tensor)
            state, distance_to_goal = self.step(state, action)
            displayed_state = state * self.magnification
            displayed_state[1] = (self.height * self.magnification
                                  - displayed_state[1])
            displayed_state = tuple(displayed_state.astype(int))
            policy_path.append(displayed_state)
            color = (0, green[i], red[i])
            cv2.line(self.image, policy_path[i], policy_path[i + 1],
                     color, thickness)
        if was_training:
            self.dqn.q_network.train()

        cv2.circle(self.image, policy_path[0], self.orb_radius,
                   self.RED, cv2.FILLED)
        cv2.circle(self.image, policy_path[-1], self.orb_radius,
                   self.GREEN, cv2.FILLED)
        return self.agent.total_reward
