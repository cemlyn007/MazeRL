import torch


class AbstractDQN(torch.nn.Module):
    HAS_TARGET_NETWORK = False

    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = self.choose_device(device)
        self.q_network = None
        self.optimizer = None
        self.loss_f = torch.nn.L1Loss(reduction='none')

    @staticmethod
    def choose_device(device) -> torch.device:
        if device:
            return device
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def train_q_network(self, transition: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        losses = self.compute_losses(transition)
        losses.sum().backward()
        self.optimizer.step()
        return losses.detach().cpu()

    def compute_losses(self, transitions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
