import torch

from abstract_dqns import dqn


class AbstractDQNWithTargetNetwork(dqn.AbstractDQN):

    def __init__(self, gamma: float = 0.9, lr: float = 0.001, device: torch.device = None):
        super().__init__(gamma=gamma, lr=lr, device=device)
        self.target_network = None

    def update_target_network(self) -> None:
        state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(state_dict)
        self.target_network.to(self.device)
        self.target_network.eval()

    def has_target_network(self) -> bool:
        return True
