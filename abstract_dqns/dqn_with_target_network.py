from abstract_dqns import dqn


class AbstractDQNWithTargetNetwork(dqn.AbstractDQN):
    HAS_TARGET_NETWORK = True

    def update_target_network(self) -> None:
        state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(state_dict)
        self.target_network.to(self.device)
        self.target_network.eval()
