import torch

from abstract_dqns import AbstractDQN
from .discrete_network import DiscreteNetwork


class DiscreteDQN(AbstractDQN):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__(gamma, lr, device)
        self.q_network = DiscreteNetwork(2, 4).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr)

    def compute_losses(self, transitions):
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        predictions = self.q_network(states).gather(1, actions).flatten()
        max_q_values = self.q_network(next_states).max(1).values
        targets = rewards + self.gamma * max_q_values
        loss = self.loss_f(predictions, targets)
        return loss
