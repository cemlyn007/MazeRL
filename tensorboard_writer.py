import torch
from torch.utils import tensorboard
from torch.utils.tensorboard import summary


class CustomSummaryWriter(tensorboard.SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, **kwargs):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = summary.hparams(hparam_dict, metric_dict)
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)
