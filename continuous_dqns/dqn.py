import torch
from torch import nn

from .network import Network
import numpy as np

TORCH_PI = torch.tensor(np.pi)


class DQN(nn.Module):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self.q_network = Network(2 + 1, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr)
        self.gamma = gamma
        self.loss_f = torch.nn.MSELoss(reduction="none")

    @staticmethod
    def has_target_network():
        return False

    def train_q_network(self, transition):
        self.optimizer.zero_grad()
        losses = self.compute_losses(transition)
        losses.sum().backward()
        self.optimizer.step()
        return losses.detach().cpu()

    @staticmethod
    def unpack(transitions):
        states = transitions[:, :2]
        actions = transitions[:, 2]
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        return states, actions, rewards, next_states

    def compute_losses(self, transitions):
        states, actions, rewards, next_states = self.unpack(transitions)
        inputs = torch.zeros((states.size(0), 3))
        max_q_values = self.get_greedy_continuous_angles(states)
        predictions = rewards + self.gamma * max_q_values
        inputs[:, :2] = states
        inputs[:, 2] = actions
        targets = self.q_network(inputs)
        loss = self.loss_f(predictions, targets)
        return loss

    def get_greedy_continuous_angles(self, states):
        max_iterations = 100
        m = 50
        n = 32
        batch_size = states.size(0)
        with torch.no_grad():
            states = states.to(self.device)
            inputs = torch.zeros((batch_size, m, 3), device=self.device)
            inputs[:, :, :2] = states
            torch.rand((batch_size, m), out=inputs[:, :, 2],
                       device=self.device)
            for iteration in range(max_iterations):
                if iteration > 0:
                    top_max_qs = q_values.topk(n, -1, sorted=False)
                    mu = top_max_qs.values.mean(-1)
                    std = top_max_qs.values.std(-1)
                    torch.normal(mu, std, size=(batch_size, m),
                                 out=inputs[:, :, 2], device=self.device)
                q_values = self.q_network(inputs.view(-1, 3))
                q_values = q_values.view((batch_size, m))
            return mu.cpu() * 2. * TORCH_PI
