from abstract_dqns import abstract_dqn


class AbstractDQNWithTargetNetwork(abstract_dqn.AbstractDQN):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__(gamma=gamma, lr=lr, device=device)
        self.target_network = None

    def update_target_network(self):
        state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(state_dict)
        self.target_network.to(self.device)
        self.target_network.eval()

    def has_target_network(self):
        return True
