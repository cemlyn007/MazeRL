import cv2
import numpy as np
import torch

import abstract_agent
import tools.abstract_graphics


class GreedyPolicyTool(tools.abstract_graphics.AbstractGraphics):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    def __init__(self, magnification: int, agent: abstract_agent.AbstractAgent,
                 max_step_num: int = 20):
        super(GreedyPolicyTool, self).__init__('Greedy Policy', magnification, agent)
        self.image = self.agent.environment.image
        self.max_step_num = max_step_num
        self.orb_radius = int(0.02 * self.agent.environment.magnification)

    def draw_blank_environment(self) -> None:
        self.agent.environment.draw_environ()
        self.agent.environment.draw_goal()
        self.image = self.agent.environment.image

    def draw(self) -> None:
        self.draw_blank_environment()
        self.draw_greedy_path()

    def draw_greedy_path(self) -> None:
        red = np.flip(np.linspace(0, 255, num=self.max_step_num + 1,
                                  endpoint=True, dtype=np.uint8)).tolist()
        green = np.linspace(0, 255, num=self.max_step_num + 1, endpoint=True,
                            dtype=np.uint8).tolist()
        thickness = int(0.004 * self.agent.environment.magnification)

        state = self.agent.environment.init_state
        policy_path = [self.convert(state)]

        was_training = False
        if self.agent.dqn.q_network.training:
            self.agent.dqn.q_network.eval()
            was_training = True

        for i in range(self.max_step_num + 1):
            with torch.no_grad():
                action = self.agent.get_greedy_action(state)
            state, distance_to_goal = self.step(state, action)
            policy_path.append(self.convert(tuple(state)))
            color = (0, green[i], red[i])
            cv2.line(self.image, policy_path[i], policy_path[i + 1],
                     color, thickness)
        if was_training:
            self.agent.dqn.q_network.train()

        cv2.circle(self.image, policy_path[0], self.orb_radius,
                   self.RED, cv2.FILLED)
        cv2.circle(self.image, policy_path[-1], self.orb_radius,
                   self.GREEN, cv2.FILLED)
        return self.agent.total_reward
