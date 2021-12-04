import torch

from abstract_dqns import double_dqn
from continuous_dqns import dqn_with_target_network


class ContinuousDoubleDQN(dqn_with_target_network.ContinuousDQNWithTargetNetwork,
                          double_dqn.AbstractDoubleDQN):
    def compute_target_q_values(self, rewards: torch.Tensor,
                                next_states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.cross_entropy_network_actions_selection(next_states,
                                                                  self.target_network)
        inputs = self.make_network_inputs(next_states, actions)
        predictions = self.q_network(inputs).squeeze(-1)
        return rewards + self.hps.gamma * predictions
