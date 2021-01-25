from abstract_dqns.stub_network import Network


class DiscreteNetwork(Network):

    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
