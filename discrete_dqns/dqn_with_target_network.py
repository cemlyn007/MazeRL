import copy

import torch

import helpers
from abstract_dqns import dqn_with_target_network
from discrete_dqns import dqn


class DiscreteDQNWithTargetNetwork(dqn_with_target_network.AbstractDQNWithTargetNetwork,
                                   dqn.DiscreteDQN):

    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device = None):
        dqn_with_target_network.AbstractDQNWithTargetNetwork.__init__(self, device)
        dqn.DiscreteDQN.__init__(self, hps, n_actions, device)
        self.target_network = copy.deepcopy(self.q_network)

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        all_q_values = self.q_network(states)
        predictions = all_q_values.gather(1, actions)
        predictions.squeeze_(1)
        targets = self.compute_q_values_using_target(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def compute_q_values_using_target(self, rewards: torch.Tensor,
                                      next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions_values = self.target_network(next_states)
        max_next_q_values, _ = next_actions_values.max(dim=1)
        return rewards + self.hps.gamma * max_next_q_values
