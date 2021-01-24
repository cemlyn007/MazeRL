import copy

from .dqn import DQN


class DQNWithTargetNetwork(DQN):

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
        next_actions_values = self.target_network(next_states).detach()
        max_next_q_values, _ = next_actions_values.max(dim=1)
        return rewards + self.gamma * max_next_q_values

    def update_target_network(self):
        state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(state_dict)
        self.target_network.to(self.device)
        self.target_network.eval()

    def has_target_network(self):
        return True
