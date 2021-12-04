import copy

import torch

import helpers
from abstract_dqns import dqn_with_target_network
from continuous_dqns import dqn


class ContinuousDQNWithTargetNetwork(dqn.ContinuousDQN,
                                     dqn_with_target_network.AbstractDQNWithTargetNetwork):

    def __init__(self, hps: helpers.Hyperparameters, device: torch.device = None):
        dqn_with_target_network.AbstractDQNWithTargetNetwork.__init__(self, device)
        dqn.ContinuousDQN.__init__(self, hps, device)
        self.target_network = copy.deepcopy(self.q_network)

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        inputs = self.make_network_inputs(states, actions)
        predictions = self.q_network(inputs)
        predictions.squeeze_(1)
        targets = self.compute_target_q_values(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def compute_target_q_values(self, rewards: torch.Tensor,
                                next_states: torch.Tensor) -> torch.Tensor:
        _, output = self.cross_entropy_network_actions_selection(next_states,
                                                                 self.target_network)
        output.mul_(self.hps.gamma)
        output.add_(rewards)
        return output
