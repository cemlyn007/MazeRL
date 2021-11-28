import numpy as np
import torch

import abstract_agent
import continuous_dqns.dqn
import environments.abstract_environment
import helpers


class ContinuousAgent(abstract_agent.AbstractAgent):

    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment,
                 dqn: continuous_dqns.dqn.ContinuousDQN, stride: float):
        super().__init__(environment, dqn)
        self.stride = stride

    def sample_angle(self, epsilon: float) -> float:
        if epsilon <= np.random.uniform():
            state = torch.from_numpy(self.state)
            angle = self._get_greedy_angle(state).item()
        else:
            angle = np.random.uniform(-np.pi, np.pi)
        return angle

    def step(self, epsilon: float = 0.) -> tuple[tuple, float]:
        angle = self.sample_angle(epsilon)
        action = self.angle_to_action(angle)
        next_state, distance_to_goal = self.environment.step(self.state, action)
        reward = self.compute_reward(distance_to_goal)
        transition = (self.state, angle, reward, next_state)
        self.state = next_state
        self.total_reward += reward
        return transition, distance_to_goal

    def _get_greedy_angle(self, state: torch.Tensor) -> torch.Tensor:
        state.unsqueeze_(0)
        state = state.to(self.dqn.device)
        angle = self.dqn.get_greedy_continuous_angles(state).cpu()
        angle.squeeze_(0)
        return angle

    @staticmethod
    def compute_reward(distance_to_goal: np.ndarray) -> np.ndarray:
        return np.square(1 - distance_to_goal).astype(distance_to_goal.dtype)

    def _get_greedy_continuous_action(self, state: torch.Tensor) -> np.ndarray:
        angle = self._get_greedy_angle(state)
        action = self.angle_to_action(angle.item())
        return action

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.from_numpy(state).to(self.dqn.device)
        state_tensor.unsqueeze_(0)
        return self._get_greedy_continuous_action(state_tensor)

    def angle_to_action(self, angle: float) -> np.ndarray:
        action = self.stride * np.array([np.cos(angle),
                                         np.sin(angle)], dtype=np.float32)
        action = helpers.trunc(action, 7)
        return action
