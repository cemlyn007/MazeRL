import copy

from abstract_dqns import AbstractDQNWithTargetNetwork
from .continuous_dqn import ContinuousDQN


class ContinuousDQNWithTargetNetwork(ContinuousDQN, AbstractDQNWithTargetNetwork):

    def __init__(self, gamma=0.9, lr=0.001, weight_decay=0.0, device=None):
        AbstractDQNWithTargetNetwork.__init__(self, gamma, lr, device)
        ContinuousDQN.__init__(self, gamma, lr, weight_decay, device)
        self.target_network = copy.deepcopy(self.q_network)

    def compute_losses(self, transitions):
        states, actions, rewards, next_states = self.unpack_transitions(transitions)
        inputs = self.make_network_inputs(states, actions)
        predictions = self.q_network(inputs).squeeze(-1)
        targets = self.compute_target_q_values(rewards, next_states)
        loss = self.loss_f(predictions, targets)
        return loss

    def compute_target_q_values(self, rewards, next_states):
        max_next_q_values = self.compute_greedy_q_values_using_target(next_states)
        return rewards + self.gamma * max_next_q_values

    def compute_greedy_q_values_using_target(self, next_states):
        _, q_values = self.cross_entropy_network_actions_selection(next_states,
                                                                   self.target_network)
        return q_values
