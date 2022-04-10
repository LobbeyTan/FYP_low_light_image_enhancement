import imp
from torch import nn
import torch


class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0) -> None:
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, is_real=True) -> torch.Tensor:
        return self.loss(prediction, self.real_label if is_real else self.fake_label)
