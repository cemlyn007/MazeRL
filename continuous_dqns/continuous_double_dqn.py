from abstract_dqns import AbstractDoubleDQN
from .continuous_dqn_with_target_network import ContinuousDQNWithTargetNetwork


class ContinuousDoubleDQN(ContinuousDQNWithTargetNetwork, AbstractDoubleDQN):

    def __init__(self, gamma=0.9, lr=0.001, weight_decay=0.0, device=None):
        super().__init__(gamma, lr, weight_decay, device)

    def compute_losses(self, transitions):
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        inputs = self.make_network_inputs(states, actions)
        predictions = self.q_network(inputs).squeeze(-1)
        targets = self.compute_target_q_values(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def compute_target_q_values(self, rewards, next_states):
        best_actions = self.compute_greedy_network_actions_using_target(next_states)
        inputs = self.make_network_inputs(next_states, best_actions)
        predictions = self.q_network(inputs).squeeze(-1)
        return rewards + self.gamma * predictions

    def compute_greedy_network_actions_using_target(self, states):
        actions, _ = self.cross_entropy_network_actions_selection(states,
                                                                  self.target_network)
        return actions
