from abstract_dqns import Network


class ContinuousNetwork(Network):

    def __init__(self, state_size, action_size):
        super().__init__(state_size + action_size, 1)
