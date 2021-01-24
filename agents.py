import numpy as np
import torch


class Agent:

    def __init__(self, environment, dqn, stride):
        self.environment = environment
        self.dqn = dqn
        self.state = None
        self.total_reward = None
        self.reset()

        self.RIGHT = np.array([stride, 0], dtype=np.float32)
        self.LEFT = np.array([-stride, 0], dtype=np.float32)
        self.UP = np.array([0, stride], dtype=np.float32)
        self.DOWN = np.array([0, -stride], dtype=np.float32)

    def reset(self):
        self.state = self.environment.reset()
        self.total_reward = 0.0

    def step(self, epsilon=0):
        if epsilon <= np.random.uniform():
            discrete_action = self.get_greedy_discrete_action(self.state)
        else:
            discrete_action = np.random.randint(0, 4)
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        next_state, distance_to_goal = self.environment.step(self.state,
                                                             continuous_action)
        reward = self._compute_reward(distance_to_goal)
        transition = (self.state, discrete_action, reward, next_state)
        self.state = next_state
        self.total_reward += reward
        return transition, distance_to_goal

    def _compute_reward(self, distance_to_goal):
        reward = (1 - distance_to_goal).astype(distance_to_goal)
        return reward

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
            raise ValueError("Unexpected value")
        return continuous_action

    def get_greedy_discrete_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.dqn.device)
            q_values_for_each_action = self.dqn.q_network(state_tensor)
            best_discrete_action = torch.argmax(q_values_for_each_action, dim=1)
        return best_discrete_action.item()
