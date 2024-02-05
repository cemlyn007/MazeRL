import numpy as np
import torch

import abstract_agent
import discrete_dqns.dqn
import environments.abstract_environment
import helpers


class DiscreteAgent(abstract_agent.AbstractAgent):

    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment,
                 dqn: discrete_dqns.dqn.DiscreteDQN, n_actions: int, stride: float):
        super().__init__(environment, dqn)
        self._n_actions = n_actions
        self._actions = self._create_actions(n_actions, stride)
        self.get_q_values = torch.compile(self.get_q_values)

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        if epsilon <= np.random.uniform():
            discrete_action = self.get_greedy_discrete_action(self.state)
        else:
            discrete_action = np.random.randint(0, self._n_actions)
        next_state, distance_to_goal = self.environment.step(self.state,
                                                             self._actions[discrete_action])
        reward = self.compute_reward(distance_to_goal)
        transition = (self.state, discrete_action, reward, next_state)
        self.state = next_state
        return transition, distance_to_goal

    def get_q_values(self, state: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=self.dqn.device,
                                        dtype=torch.float32)
            state_tensor.unsqueeze_(0)
            q_values = self.dqn.q_network(state_tensor).cpu()
        q_values.squeeze_(0)
        return q_values
    
    def get_batch_q_values(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            states = states.to(self.dqn.device)
            q_values = self.dqn.q_network(states).cpu()
        q_values.squeeze_(0)
        return q_values

    def get_greedy_discrete_action(self, state: np.ndarray) -> int:
        return self.get_q_values(state).argmax().item()

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        discrete_action = self.get_greedy_discrete_action(state)
        return self._actions[discrete_action]

    def _create_actions(self, n_actions: int, stride: float) -> list[np.ndarray]:
        actions = []
        for i in range(n_actions):
            theta = 2. * np.pi * i / n_actions
            action = stride * np.array([np.cos(theta), np.sin(theta)])
            actions.append(helpers.trunc(action, 5))
        return actions
