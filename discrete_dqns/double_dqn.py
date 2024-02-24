import torch

from discrete_dqns import dqn_with_target_network


class DiscreteDoubleDQN(dqn_with_target_network.DiscreteDQNWithTargetNetwork):
    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, dones, next_states = self._unpack_transitions(transitions)
        all_q_values = self.q_network(states)
        predicted_q_values = all_q_values.gather(1, actions).flatten(0)
        target_q_values = self._compute_q_values_using_target(rewards, dones, next_states)
        loss = self.loss_f(predicted_q_values, target_q_values)
        return loss

    def _compute_q_values_using_target(self, rewards: torch.Tensor,
                                       dones: torch.Tensor,
                                       next_states: torch.Tensor) -> torch.Tensor:
        # TODO: Wrong way round: https://arxiv.org/pdf/1509.06461.pdf
        q_values = self.q_network(next_states)
        target_q_values = self.target_network(next_states).detach()
        best_discrete_actions = torch.argmax(target_q_values, 1, False)
        best_discrete_actions.unsqueeze_(-1)
        predicted_q_values = torch.gather(q_values, 1, best_discrete_actions)
        predicted_q_values.squeeze_(-1)
        return rewards + self.hps.gamma * predicted_q_values * (1 - dones)
