import copy

import torch

import helpers
from discrete_dqns import dqn
from discrete_dqns import network

class DiscreteDQNWithTargetNetwork(dqn.DiscreteDQN):
    def __init__(self, hps: helpers.Hyperparameters, n_actions: int, device: torch.device):
        super().__init__(hps, n_actions, device)
        self.target_network = copy.deepcopy(self.q_network)

    @property
    def has_target_network(self) -> bool:
        return True

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self._unpack_transitions(transitions)
        all_q_values = self.q_network(states)
        predictions = all_q_values.gather(1, actions)
        predictions.squeeze_(1)
        targets = self._compute_q_values_using_target(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def _compute_q_values_using_target(self, rewards: torch.Tensor,
                                       next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions_values = self.target_network(next_states)
        max_next_q_values, _ = next_actions_values.max(dim=1)
        return rewards + self.hps.gamma * max_next_q_values

    def update_target_network(self) -> None:
        state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(state_dict)
        self.target_network.to(self.device)
        self.target_network.eval()
