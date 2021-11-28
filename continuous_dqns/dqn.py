import numpy as np
import torch

from abstract_dqns import dqn
from abstract_dqns import stub_network

TORCH_PI = torch.tensor(np.pi)
TORCH_I = torch.tensor(np.complex(0, 1))


class ContinuousDQN(dqn.AbstractDQN):

    def __init__(self, gamma: float = 0.9, lr: float = 0.001, weight_decay: float = 0.,
                 device: torch.device = None):
        super().__init__(gamma, lr, device)
        self.weight_decay = weight_decay
        self.q_network = stub_network.Network(2 + 1, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),
                                          lr=self.lr,
                                          betas=(0.9, 0.975), eps=0.1,
                                          weight_decay=self.weight_decay,
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
        max_q_values = self.get_greedy_continuous_q_values(next_states)
        targets = rewards + self.gamma * max_q_values
        loss = self.loss_f(predictions, targets)
        return loss

    def make_network_inputs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.empty((states.size(0), 3), device=self.device)
        inputs[:, :2] = states
        inputs[:, 2] = self.angles_to_network_actions(actions)
        return inputs

    def cross_entropy_network_actions_selection(self, states: torch.Tensor,
                                                network: torch.nn.Module
                                                ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.cross_entropy_m
        n = self.cross_entropy_n
        batch_size = states.size(0)
        with torch.no_grad():
            states = states.to(self.device)
            inputs = torch.empty((batch_size, m, 3), device=self.device)
            inputs[:, :, :2] = states.unsqueeze(1)
            sampled_angles = inputs[:, :, 2].clone()

            flat_inputs = inputs.view(-1, 3)

            torch.rand((batch_size, m), out=sampled_angles, device=self.device)
            sampled_angles.sub_(0.5)
            sampled_angles.mul_(TORCH_PI)

            mu = torch.empty((batch_size, 1), device=self.device)
            std = torch.empty((batch_size, 1), device=self.device)
            q_values = network(inputs.view(-1, 3)).view((batch_size, m))
            sampled_network_actions = inputs[:, :, 2]

            sins = torch.empty((batch_size, n), device=self.device)
            coss = torch.empty_like(sins)

            Y = torch.empty((batch_size, 1), device=self.device)
            X = torch.empty_like(Y)

            top_k_angles = torch.empty((batch_size, n), device=self.device)
            casted_mu, casted_std, _ = torch.broadcast_tensors(mu, std, sampled_angles)
            for iteration in range(self.cross_entropy_max_iters):
                top_k_q_values, top_k_q_indices = q_values.topk(n, -1, sorted=False)
                torch.gather(sampled_angles, -1, top_k_q_indices, out=top_k_angles)

                torch.sin(top_k_angles, out=sins)
                torch.cos(top_k_angles, out=coss)
                sins.mul_(1. / m)
                coss.mul_(1. / m)
                torch.sum(sins, -1, keepdim=True, out=Y)
                torch.sum(coss, -1, keepdim=True, out=X)
                torch.atan2(Y, X, out=mu)

                Y.square_()
                X.square_()
                Y.add_(X)
                torch.sqrt(Y, out=std)

                torch.normal(mean=casted_mu, std=casted_std, out=sampled_angles)
                sampled_network_actions[:] = self.angles_to_network_actions(sampled_angles)
                q_values = network(flat_inputs).view((batch_size, m))

            return sampled_network_actions.mean(-1), q_values.mean(-1)

    @staticmethod
    def angles_to_network_actions(angles: torch.Tensor) -> torch.Tensor:
        return angles.div(TORCH_PI)  # output ranges(-1,+1)

    @staticmethod
    def network_actions_to_angles(network_inputs: torch.Tensor) -> torch.Tensor:
        return network_inputs.mul(TORCH_PI)  # output ranges(-pi,+pi)

    def get_greedy_continuous_q_values(self, states: torch.Tensor) -> torch.Tensor:
        _, values = self.cross_entropy_network_actions_selection(states, self.q_network)
        return values

    def get_greedy_continuous_angles(self, states: torch.Tensor) -> torch.Tensor:
        network_actions, _ = self.cross_entropy_network_actions_selection(states, self.q_network)
        return self.network_actions_to_angles(network_actions)

    def get_greedy_continuous_network_actions(self, states: torch.Tensor) -> torch.Tensor:
        network_actions, _ = self.cross_entropy_network_actions_selection(states, self.q_network)
        return network_actions
