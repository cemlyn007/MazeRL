import torch
from torch import nn

from .network import Network


class DQN(nn.Module):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        self.q_network = Network(2, 4).to(self.device)
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

    def compute_losses(self, transitions):
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        max_q_values = self.q_network(next_states).max(1)[0]
        predictions = rewards + self.gamma * max_q_values
        targets = self.q_network(states).gather(1, actions).flatten()
        loss = self.loss_f(predictions, targets)
        return loss
