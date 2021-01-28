import torch

from .discrete_dqn_with_target_network import DiscreteDQNWithTargetNetwork


class DiscreteDoubleDQN(DiscreteDQNWithTargetNetwork):

    def __init__(self, gamma=0.9, lr=0.001, weight_decay=0.0, device=None):
        super().__init__(gamma, lr, weight_decay, device)

    def compute_losses(self, transitions):
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        all_q_values = self.q_network(states)
        predicted_q_values = all_q_values.gather(1, actions).flatten(0)
        target_q_values = self.compute_q_values_using_target(rewards, next_states)
        loss = self.loss_f(predicted_q_values, target_q_values)
        return loss

    def compute_q_values_using_target(self, rewards, next_states):
        q_values = self.q_network(next_states)
        target_q_values = self.target_network(next_states).detach()
        best_discrete_actions = torch.argmax(target_q_values, 1, False)
        best_discrete_actions.unsqueeze_(-1)
        predicted_q_values = torch.gather(q_values, 1, best_discrete_actions)
        predicted_q_values.squeeze_(-1)
        return rewards + self.gamma * predicted_q_values
