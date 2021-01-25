import torch

from abstract_dqns import AbstractDQNWithTargetNetwork
from .discrete_dqn import DiscreteDQN


class DiscreteDQNWithTargetNetwork(AbstractDQNWithTargetNetwork, DiscreteDQN):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__(gamma, lr, device)

    def compute_losses(self, transitions):
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        all_q_values = self.q_network(states)
        predictions = all_q_values.gather(1, actions).flatten(0)
        targets = self.compute_q_values_using_target(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def compute_q_values_using_target(self, rewards, next_states):
        with torch.no_grad():
            next_actions_values = self.target_network(next_states)
        max_next_q_values, _ = next_actions_values.max(dim=1)
        return rewards + self.gamma * max_next_q_values
