import copy

import torch

import helpers
from continuous_dqns import dqn
from abstract_dqns import stub_network


class ContinuousDQNWithTargetNetwork(dqn.ContinuousDQN):

    def __init__(self, hps: helpers.Hyperparameters, device: torch.device):
        super().__init__(hps, device)
        self.loss_f = torch.nn.L1Loss(reduction='none')
        self.q_network = stub_network.Network(2 + 1, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.q_network.parameters(),
                                         lr=hps.lr)
        self.target_network = copy.deepcopy(self.q_network)

    @property
    def has_target_network(self) -> bool:
        return True

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self._unpack_transitions(transitions)
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

    def update_target_network(self) -> None:
        state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(state_dict)
        self.target_network.to(self.device)
        self.target_network.eval()