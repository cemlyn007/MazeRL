import numpy as np
import torch

import helpers
from abstract_dqns import dqn
from abstract_dqns import stub_network

TORCH_PI = torch.tensor(np.pi)
TORCH_I = torch.tensor(np.complex(0, 1))


class ContinuousDQN(dqn.AbstractDQN):

    def __init__(self, hps: helpers.Hyperparameters, device: torch.device = None):
        super().__init__(device)
        self.hps = hps
        self.q_network = stub_network.Network(2 + 1, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),
                                          lr=hps.lr, weight_decay=hps.weight_decay,
                                          amsgrad=True)
        self.cross_entropy_max_iters = 16
        self.cross_entropy_m = 64
        self.cross_entropy_n = 12

    @staticmethod
    def unpack_transitions(transitions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
        states = transitions[:, :2]
        actions = transitions[:, 2]
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        return states, actions, rewards, next_states

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        inputs = self.make_network_inputs(states, actions)
        predictions = self.q_network(inputs).squeeze(-1)
        _, max_q_values = self.cross_entropy_network_actions_selection(states, self.q_network)
        targets = rewards + self.hps.gamma * max_q_values
        loss = self.loss_f(predictions, targets)
        return loss

    def make_network_inputs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.empty((states.size(0), 3), device=self.device)
        inputs[:, :2] = states
        inputs[:, 2] = self.angles_to_network_actions(actions)
        return inputs

    def _sample_uniform_network_actions_(self, t: torch.Tensor) -> None:
        torch.rand(t.shape, out=t, device=self.device)
        t.mul_(2.)
        t.sub_(1.)

    def cross_entropy_network_actions_selection(self, states: torch.Tensor,
                                                network: torch.nn.Module
                                                ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.cross_entropy_m  # angle sample size per state.
        n = self.cross_entropy_n  # top k.
        batch_size = states.size(0)
        with torch.no_grad():
            states = states.to(self.device)
            inputs = torch.empty((batch_size, m, 3), device=self.device)
            inputs[:, :, :2] = states.unsqueeze(1)
            self._sample_uniform_network_actions_(inputs[:, :, 2])
            sampled_angles = torch.pi * inputs[:, :, 2]
            flat_inputs = inputs.view(-1, 3)
            mu = torch.empty((batch_size, 1), device=self.device)
            std = torch.empty((batch_size, 1), device=self.device)
            sins = torch.empty((batch_size, n), device=self.device)
            coss = torch.empty_like(sins)
            Y = torch.empty((batch_size, 1), device=self.device)
            X = torch.empty_like(Y)
            top_k_angles = torch.empty((batch_size, n), device=self.device)
            casted_mu, casted_std, _ = torch.broadcast_tensors(mu, std, sampled_angles)
            for _ in range(self.cross_entropy_max_iters):
                q_values = network(flat_inputs).view((batch_size, m))
                top_k_q_values, top_k_q_indices = q_values.topk(n, -1, sorted=False)
                torch.gather(sampled_angles, -1, top_k_q_indices, out=top_k_angles)

                torch.sin(top_k_angles, out=sins)
                torch.cos(top_k_angles, out=coss)
                torch.sum(sins, -1, keepdim=True, out=Y)
                torch.sum(coss, -1, keepdim=True, out=X)
                torch.atan2(Y, X, out=mu)

                Y.square_()
                X.square_()
                Y.add_(X)
                torch.sqrt(Y, out=std)

                torch.normal(mean=casted_mu, std=casted_std, out=sampled_angles)
                inputs[:, :, 2] = self.angles_to_network_actions(sampled_angles)

            mu.squeeze_(1)
            return mu, top_k_q_values.mean(-1)

    @staticmethod
    def angles_to_network_actions(angles: torch.Tensor) -> torch.Tensor:
        actions = angles.remainder(2 * torch.pi)
        actions.sub_(torch.pi)
        actions.div_(torch.pi)
        return actions  # output ranges(-1,+1)

    def get_greedy_continuous_angles(self, states: torch.Tensor) -> torch.Tensor:
        angles, _ = self.cross_entropy_network_actions_selection(states, self.q_network)
        return angles
