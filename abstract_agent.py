import numpy as np

import abstract_dqns.dqn
import environments.abstract_environment


class AbstractAgent:

    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment,
                 dqn: abstract_dqns.dqn.AbstractDQN):
        self.environment = environment
        self.dqn = dqn
        self.state = None
        self.total_reward = None
        self.reset()

    def reset(self) -> None:
        self.state = self.environment.reset()
        self.total_reward = 0.0

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        raise NotImplementedError

    @staticmethod
    def compute_reward(distance_to_goal: np.ndarray) -> float:
        return -distance_to_goal

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
