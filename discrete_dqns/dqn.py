import torch

import helpers
from abstract_dqns import dqn
from discrete_dqns import network


class DiscreteDQN(dqn.AbstractDQN):

    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device = None):
        super().__init__(device)
        self.hps = hps
        self.q_network = network.DiscreteNetwork(2, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), hps.lr,
                                          weight_decay=hps.weight_decay)
        self.loss_f = torch.nn.MSELoss(reduction='none')

    @staticmethod
    def unpack_transitions(transitions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        return states, actions, rewards, next_states

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        predictions = self.q_network(states).gather(1, actions).flatten()
        max_q_values = self.q_network(next_states).max(1).values
        targets = rewards + self.hps.gamma * max_q_values
        loss = self.loss_f(predictions, targets)
        return loss
