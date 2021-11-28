import torch

from abstract_dqns import dqn_with_target_network


class AbstractDoubleDQN(dqn_with_target_network.AbstractDQNWithTargetNetwork):

    def __init__(self, gamma: float = 0.9, lr: float = 0.001, device: torch.device = None):
        super().__init__(gamma, lr, device)
