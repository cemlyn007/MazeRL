import copy
import torch
from .dqn_with_target_network import DQNWithTargetNetwork


class DoubleDQN(DQNWithTargetNetwork):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__(gamma, lr, device)
        self.target_network = copy.deepcopy(self.q_network)

    def compute_losses(self, transitions):
        states = transitions[:, :2]
        actions = transitions[:, 2].long().unsqueeze(-1)
        rewards = transitions[:, 3]
        next_states = transitions[:, 4:]
        all_q_values = self.q_network(states)
        predicted_q_values = all_q_values.gather(1, actions).flatten(0)
        target_q_values = self.compute_target_q_values(rewards, next_states)
        loss = self.loss_f(predicted_q_values, target_q_values)
        return loss

    def compute_target_q_values(self, rewards, next_states):
        q_values = self.q_network(next_states)
        target_q_values = self.target_network(next_states).detach()
        best_discrete_actions = torch.argmax(target_q_values, 1, False)
        best_discrete_actions.unsqueeze_(-1)
        predicted_q_values = torch.gather(q_values, 1, best_discrete_actions)
        predicted_q_values.squeeze_(-1)
        return rewards + self.gamma * predicted_q_values
