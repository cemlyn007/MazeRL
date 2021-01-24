import numpy as np
import torch


class Agent:

    def __init__(self, environment, dqn, stride):
        self.environment = environment
        self.dqn = dqn
        self.stride = stride
        self.state = None
        self.total_reward = None
        self.reset()

    def reset(self):
        self.state = self.environment.reset()
        self.total_reward = 0.0

    def step(self, epsilon=0):
        if epsilon <= np.random.uniform():
            angle = self.get_greedy_continuous_angle(self.state)
        else:
            angle = np.random.uniform(0, 1) * 2 * np.pi
        action = self.stride * np.array([np.cos(angle),
                                         np.sin(angle)],
                                        dtype=np.float32)
        next_state, distance_to_goal = self.environment.step(self.state,
                                                             action)
        reward = self._compute_reward(distance_to_goal)
        transition = (self.state, angle, reward, next_state)
        self.state = next_state
        self.total_reward += reward
        return transition, distance_to_goal

    @staticmethod
    def _compute_reward(distance_to_goal):
        reward = (1 - distance_to_goal).astype(distance_to_goal)
        return reward

    def get_greedy_continuous_angle(self, state):
        max_iterations = 100
        m = 50
        n = 32
        with torch.no_grad():
            state_tensor = torch.tensor(state).to(self.dqn.device)
            inputs = torch.zeros((m, 3), device=self.dqn.device)
            inputs[:, :2] = state_tensor
            torch.rand(m, out=inputs[:, 2], device=self.dqn.device)
            for iteration in range(max_iterations):
                if iteration > 0:
                    top_max_qs = q_values.topk(n, 0)
                    mu = top_max_qs.values.mean()
                    std = top_max_qs.values.std()
                    torch.normal(mu, std, size=m, out=inputs[:, 2],
                                 device=self.dqn.device)
                q_values = self.dqn.q_network(inputs)
            return mu.cpu().numpy() * 2 * np.pi

