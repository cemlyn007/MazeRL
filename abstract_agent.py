import numpy as np
import torch
import environments.abstract_environment


class AbstractAgent:

    def __init__(self, environment: environments.abstract_environment.AbstractEnvironment):
        self.environment = environment
        self.state = None
        self.reset()

    def reset(self) -> None:
        self.state = self.environment.reset()

    def step(self, epsilon: float = 0) -> tuple[tuple, float]:
        raise NotImplementedError

    @staticmethod
    def compute_reward(distance_to_goal: np.ndarray) -> float:
        return -distance_to_goal

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_batch_q_values(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
