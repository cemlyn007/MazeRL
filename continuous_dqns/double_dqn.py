import torch

from abstract_dqns import double_dqn
from continuous_dqns import dqn_with_target_network


class ContinuousDoubleDQN(dqn_with_target_network.ContinuousDQNWithTargetNetwork,
                          double_dqn.AbstractDoubleDQN):

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        inputs = self.make_network_inputs(states, actions)
        predictions = self.q_network(inputs).squeeze(-1)
        targets = self.compute_target_q_values(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def compute_target_q_values(self, rewards: torch.Tensor,
                                next_states: torch.Tensor) -> torch.Tensor:
        best_actions = self.compute_greedy_network_actions_using_target(next_states)
        inputs = self.make_network_inputs(next_states, best_actions)
        predictions = self.q_network(inputs).squeeze(-1)
        return rewards + self.hps.gamma * predictions

    def compute_greedy_network_actions_using_target(self, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.cross_entropy_network_actions_selection(states, self.target_network)
        return actions
