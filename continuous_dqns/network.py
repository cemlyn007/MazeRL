from abstract_dqns import stub_network


class ContinuousNetwork(stub_network.Network):

    def __init__(self, state_size, action_size):
        super().__init__(state_size + action_size, 1)
