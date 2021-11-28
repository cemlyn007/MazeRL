import copy

import torch

from abstract_dqns import dqn_with_target_network
from discrete_dqns import dqn


class DiscreteDQNWithTargetNetwork(dqn_with_target_network.AbstractDQNWithTargetNetwork,
                                   dqn.DiscreteDQN):

    def __init__(self, gamma=0.9, lr=0.001, weight_decay=0.0, device=None):
        dqn_with_target_network.AbstractDQNWithTargetNetwork.__init__(self,
                                                                               gamma=gamma, lr=lr,
                                                                               device=device)
        dqn.DiscreteDQN.__init__(self, gamma=gamma, lr=lr,
                                          weight_decay=weight_decay, device=device)
        self.target_network = copy.deepcopy(self.q_network)

    def compute_losses(self, transitions):
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
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
