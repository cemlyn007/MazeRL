import torch

from abstract_dqns import dqn
from discrete_dqns import network


class DiscreteDQN(dqn.AbstractDQN):

    def __init__(self, gamma=0.9, lr=0.001, weight_decay=0.0, device=None):
        super().__init__(gamma, lr, device)
        self.weight_decay = weight_decay
        self.q_network = network.DiscreteNetwork(2, 4).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.lr,
                                          weight_decay=self.weight_decay)
        self.loss_f = torch.nn.MSELoss(reduction='none')

    @staticmethod
    def unpack_transitions(transitions):
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        return states, actions, rewards, next_states

    def compute_losses(self, transitions):
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        predictions = self.q_network(states).gather(1, actions).flatten()
        max_q_values = self.q_network(next_states).max(1).values
        targets = rewards + self.gamma * max_q_values
        loss = self.loss_f(predictions, targets)
        return loss
