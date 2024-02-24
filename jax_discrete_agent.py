import numpy as np
import torch

import abstract_agent
import jax.numpy as jnp
import environments.abstract_environment
import helpers
import jax_discrete_dqns.dqn

class DiscreteAgent(abstract_agent.AbstractAgent):
    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment,
                 dqn: jax_discrete_dqns.dqn.DiscreteDQN, n_actions: int, stride: float):
        super().__init__(environment)
        self.dqn = dqn
        self._n_actions = n_actions
        self._actions = self._create_actions(n_actions, stride)

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        if epsilon <= np.random.uniform():
            discrete_action = self.get_batch_q_values(
                torch.tensor(self.state, dtype=torch.float32).unsqueeze_(0)
            ).argmax().item()
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

    def get_batch_q_values(self, states: torch.Tensor) -> torch.Tensor:
        q_values = self.dqn.q_network.apply(self.dqn._params, jnp.array(states))
        return torch.from_numpy(np.array(q_values))

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        discrete_action = self.get_batch_q_values(
            torch.tensor(state, dtype=torch.float32).unsqueeze_(0)
        ).argmax().item()
        return self._actions[discrete_action]
 
    def _create_actions(self, n_actions: int, stride: float) -> list[np.ndarray]:
        actions = []
        for i in range(n_actions):
            theta = 2. * np.pi * i / n_actions
            action = stride * np.array([np.cos(theta), np.sin(theta)])
            actions.append(helpers.trunc(action, 5))
        return actions
