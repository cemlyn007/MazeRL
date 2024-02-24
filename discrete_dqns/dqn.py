import torch

import helpers
from abstract_dqns import dqn
from discrete_dqns import network


class DiscreteDQN(dqn.AbstractDQN, torch.nn.Module):
    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device):
        super().__init__()
        self.device = device
        self.hps = hps
        self.q_network = network.DiscreteNetwork(2, n_actions).to(self.device)
        self.optimizer = torch.optim.SGD(self.q_network.parameters(), hps.lr)
        self.loss_f = torch.nn.L1Loss(reduction='none')

    @property
    def has_target_network(self) -> bool:
        return False

    def train_q_network(self, transition: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        losses = self.compute_losses(transition)
        losses.sum().backward()
        self.optimizer.step()
        return losses.detach().cpu()

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        predictions = self.q_network(states).gather(1, actions).flatten()
        max_q_values = self.q_network(next_states).max(1).values
        targets = rewards + self.hps.gamma * max_q_values * (1 - dones)
        loss = self.loss_f(predictions, targets)
        return loss

    @staticmethod
    def _unpack_transitions(transitions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor, torch.Tensor]:
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        dones = transitions[:, 4]
        next_states = transitions[:, 5:]
        return states, actions, rewards, dones, next_states
