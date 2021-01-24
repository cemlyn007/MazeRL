import torch


class Network(torch.nn.Module):

    def __init__(self, state_size, action_size):
        super(Network, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, action_size)
        )

    def forward(self, states):
        return self.model(states)
