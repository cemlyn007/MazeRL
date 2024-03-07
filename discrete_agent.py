import numpy as np
import torch

import abstract_agent
import discrete_dqns.dqn
import environments.abstract_environment
import helpers


class DiscreteAgent(abstract_agent.AbstractAgent):
    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment,
                 dqn: discrete_dqns.dqn.DiscreteDQN, n_actions: int, stride: float) -> None:
        super().__init__(environment)
        self.dqn = dqn
        self._n_actions = n_actions
        self._actions = self._create_actions(n_actions, stride)
        self._get_batch_q_values = torch.compile(self._get_batch_q_values, disable=self.dqn.device is torch.device("cpu"))
        self.dqn.train_q_network = torch.compile(self.dqn.train_q_network)

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        if epsilon <= np.random.uniform():
            discrete_action = self.get_batch_q_values(np.expand_dims(self.state, 0)).argmax().item()
        else:
            discrete_action = np.random.randint(0, self._n_actions)
        next_state, distance_to_goal = self.environment.step(
            self.state,
            self._actions[discrete_action]
        )
        reward = self.compute_reward(distance_to_goal)
        transition = (self.state, discrete_action, reward, next_state)
        self.state = next_state
        return transition, distance_to_goal

    def get_batch_q_values(self, states: np.ndarray) -> np.ndarray:
        return self._get_batch_q_values(torch.tensor(states, dtype=torch.float32)).numpy()
    
    def _get_batch_q_values(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.dqn.q_network(states)
        return q_values

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        discrete_action = self.get_batch_q_values(np.expand_dims(state, 0)).argmax().item()
        return self._actions[discrete_action]
 
    def _create_actions(self, n_actions: int, stride: float) -> np.ndarray:
        actions = []
        for i in range(n_actions):
            theta = 2. * np.pi * i / n_actions
            action = stride * np.array([np.cos(theta), np.sin(theta)])
            actions.append(helpers.trunc(action, 5))
        return np.stack(actions)
