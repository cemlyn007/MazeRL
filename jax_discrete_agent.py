import numpy as np
import jax

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
        self._get_batch_q_values = jax.jit(self._get_batch_q_values)

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        if epsilon <= np.random.uniform():
            discrete_action = self.get_batch_q_values(self.state).argmax()
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
        q_values = self._get_batch_q_values(self.dqn._params, jnp.asarray(states))
        return np.asarray(q_values)

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        discrete_action = self._get_batch_q_values(self.dqn._params, jnp.expand_dims(jnp.asarray(state), 0)).item()
        return self._actions[discrete_action]
    
    def _get_batch_q_values(self, params, states: jax.Array) -> jax.Array:
        q_values = self.dqn.q_network.apply(params, states)
        return q_values
 
    def _create_actions(self, n_actions: int, stride: float) -> np.ndarray:
        actions = []
        for i in range(n_actions):
            theta = 2. * np.pi * i / n_actions
            action = stride * np.array([np.cos(theta), np.sin(theta)])
            actions.append(helpers.trunc(action, 5))
        return np.stack(actions)
