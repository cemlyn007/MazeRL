import numpy as np
import torch

import abstract_agent


class DiscreteAgent(abstract_agent.AbstractAgent):

    def __init__(self, environment, dqn, stride):
        super().__init__(environment, dqn)
        self.RIGHT = np.array([stride, 0], dtype=np.float32)
        self.LEFT = np.array([-stride, 0], dtype=np.float32)
        self.UP = np.array([0, stride], dtype=np.float32)
        self.DOWN = np.array([0, -stride], dtype=np.float32)

    def step(self, epsilon=0):
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

    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = self.RIGHT
        elif discrete_action == 1:  # Move left
            continuous_action = self.LEFT
        elif discrete_action == 2:  # Move up
            continuous_action = self.UP
        elif discrete_action == 3:  # Move down
            continuous_action = self.DOWN
        else:
            raise ValueError('Unexpected value')
        return continuous_action

    def get_greedy_discrete_action(self, state):
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                state_tensor = state.to(self.dqn.device)
                state_tensor.unsqueeze_(0)
            else:
                state_tensor = torch.tensor(state,
                                            device=self.dqn.device).unsqueeze(0)
            q_values_for_each_action = self.dqn.q_network(state_tensor)
            best_discrete_action = torch.argmax(q_values_for_each_action, dim=1)
        return best_discrete_action.item()

    def get_greedy_action(self, state):
        discrete_action = self.get_greedy_discrete_action(state)
        return self._discrete_action_to_continuous(discrete_action)
