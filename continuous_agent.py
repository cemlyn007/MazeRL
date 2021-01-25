import numpy as np
import torch

import helpers
from abstract_agent import AbstractAgent


class ContinuousAgent(AbstractAgent):

    def __init__(self, environment, dqn, stride):
        super().__init__(environment, dqn)
        self.stride = stride

    def sample_angle(self, epsilon):
        if epsilon <= np.random.uniform():
            state = torch.from_numpy(self.state)
            angle = self._get_greedy_angle(state)
        else:
            angle = self.get_random_angle()
        return angle

    def step(self, epsilon=0):
        angle = self.sample_angle(epsilon)
        action = self.angle_to_action(angle)
        next_state, distance_to_goal = self.environment.step(self.state, action)
        reward = self.compute_reward(distance_to_goal)
        transition = (self.state, angle, reward, next_state)
        self.state = next_state
        self.total_reward += reward
        return transition, distance_to_goal

    def _get_greedy_angle(self, state: torch.Tensor):
        state.unsqueeze_(0)
        state = state.to(self.dqn.device)
        angle = self.dqn.get_greedy_continuous_angles(state).cpu()
        angle.squeeze_(0)
        return angle

    @staticmethod
    def compute_reward(distance_to_goal):
        return np.square(1 - distance_to_goal).astype(distance_to_goal.dtype)

    @staticmethod
    def get_random_angle():
        return np.random.uniform(-np.pi, np.pi)

    def _get_greedy_continuous_action(self, state: torch.Tensor):
        angle = self._get_greedy_angle(state)
        action = self.angle_to_action(angle)
        return action

    def get_greedy_action(self, state: torch.Tensor):
        state = state.to(self.dqn.device)
        action = self._get_greedy_continuous_action(state)
        return action

    def angle_to_action(self, angle):
        action = self.stride * np.array([np.cos(angle),
                                         np.sin(angle)], dtype=np.float32)
        action = helpers.trunc(action, 7)
        return action
