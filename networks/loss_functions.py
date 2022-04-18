import imp
from torch import nn
import torch


class GANLoss(nn.Module):

    def __init__(self, gan_mode="lsgan", target_real_label=1.0, target_fake_label=0.0) -> None:
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, is_real=True) -> torch.Tensor:
        target_tensor = self.real_label if is_real else self.fake_label
        return self.loss(prediction, target_tensor.expand_as(prediction))
