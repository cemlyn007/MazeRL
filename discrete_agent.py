import numpy as np
import torch

import abstract_agent
import discrete_dqns.dqn
import environments.abstract_environment


class DiscreteAgent(abstract_agent.AbstractAgent):

    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment,
                 dqn: discrete_dqns.dqn.DiscreteDQN, stride: float):
        super().__init__(environment, dqn)
        self.RIGHT = np.array([stride, 0], dtype=np.float32)
        self.LEFT = np.array([-stride, 0], dtype=np.float32)
        self.UP = np.array([0, stride], dtype=np.float32)
        self.DOWN = np.array([0, -stride], dtype=np.float32)

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        if epsilon <= np.random.uniform():
            discrete_action = self.get_greedy_discrete_action(self.state)
        else:
            discrete_action = np.random.randint(0, 4)
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        next_state, distance_to_goal = self.environment.step(self.state,
                                                             continuous_action)
        reward = self.compute_reward(distance_to_goal)
        transition = (self.state, discrete_action, reward, next_state)
        self.state = next_state
        self.total_reward += reward
        return transition, distance_to_goal

    def _discrete_action_to_continuous(self, discrete_action: int) -> np.ndarray:
        if discrete_action == 0:
            continuous_action = self.RIGHT
        elif discrete_action == 1:
            continuous_action = self.UP
        elif discrete_action == 2:
            continuous_action = self.LEFT
        elif discrete_action == 3:
            continuous_action = self.DOWN
        else:
            raise ValueError('Unexpected value')
        return continuous_action

    def get_q_values(self, state: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=self.dqn.device,
                                        dtype=torch.float32)
            state_tensor.unsqueeze_(0)
            q_values = self.dqn.q_network(state_tensor).cpu()
        q_values.squeeze_(0)
        return q_values

    def get_greedy_discrete_action(self, state: np.ndarray) -> int:
        return self.get_q_values(state).argmax().item()

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        discrete_action = self.get_greedy_discrete_action(state)
        return self._discrete_action_to_continuous(discrete_action)
