from abstract_dqns import abstract_dqn_with_target_network


class AbstractDoubleDQN(abstract_dqn_with_target_network.AbstractDQNWithTargetNetwork):

    def __init__(self, gamma=0.9, lr=0.001, device=None):
        super().__init__(gamma, lr, device)
