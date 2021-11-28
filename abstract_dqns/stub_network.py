import torch


class Network(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(Network, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, output_size)
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states)
