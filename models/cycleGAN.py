
import itertools
import os
import torch
from networks.discriminators import NLayerDiscriminator
from networks.generators import ResnetGenerator
from networks.loss_functions import GANLoss
from torch import nn


class CycleGANModel:

    def __init__(self, lr=0.001, lamda_A=10.0, lamda_B=10.0) -> None:
        self.lambda_A = torch.tensor(lamda_A)
        self.lambda_B = torch.tensor(lamda_B)

        self.G_X = ResnetGenerator(3, 3)
        self.D_Y = NLayerDiscriminator(3)

        self.F_Y = ResnetGenerator(3, 3)
        self.D_X = NLayerDiscriminator(3)

        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(
            self.G_X.parameters(), self.F_Y.parameters()), lr=lr)

        self.optimizer_D = torch.optim.Adam(itertools.chain(
            self.D_X.parameters(), self.D_Y.parameters()), lr=lr)

    def forward(self, real_X, real_Y):
        self.real_X = real_X
        self.real_Y = real_Y

        self.fake_Y = self.G_X(real_X)
        self.fake_X = self.F_Y(real_Y)

        self.reconstruc_X = self.F_Y(self.fake_Y)
        self.reconstruc_Y = self.G_X(self.fake_X)

    def _backward_D(self, D: nn.Module, real: torch.Tensor, fake: torch.Tensor):

        real_D_loss = self.criterionGAN(D(real), is_real=True)
        fake_D_loss = self.criterionGAN(D(fake.detach()), is_real=False)

        # To slows down the rate at which D learns relative to G
        loss_D = (fake_D_loss + real_D_loss) * 0.5

        loss_D.backward()

        return loss_D

    def backward_D_X(self):
        self.loss_D_X = self._backward_D(self.D_X, self.real_X, self.fake_X)

    def backward_D_Y(self):
        self.loss_D_Y = self._backward_D(self.D_Y, self.real_Y, self.fake_Y)

    def backward_G(self):

        self.loss_G_X = self.criterionGAN(self.D_Y(self.fake_Y), is_real=True)
        self.loss_F_Y = self.criterionGAN(self.D_X(self.fake_X), is_real=True)

        self.loss_cycle_X = self.criterionCycle(
            self.reconstruc_X, self.real_X) * self.lambda_A

        self.loss_cycle_Y = self.criterionCycle(
            self.reconstruc_Y, self.real_Y) * self.lambda_B

        self.loss_G = self.loss_G_X + self.loss_F_Y + \
            self.loss_cycle_X + self.loss_cycle_Y

        self.loss_G.backward()

    def optimize_parameters(self, real_X, real_Y):
        # forward
        self.forward(real_X, real_Y)

        # Update G_X & F_Y #

        # Stop update on Discriminators
        self.D_X.requires_grad_(False)
        self.D_Y.requires_grad_(False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Update D_X& D_Y # TODO(why no stop update on Generators, is because of detach?)
        self.D_X.requires_grad_(True)
        self.D_Y.requires_grad_(True)
        self.optimizer_D.zero_grad()
        self.backward_D_X()
        self.backward_D_Y()
        self.optimizer_D.step()

    def save_model(self, directory, epoch):
        self._save_network(self.G_X, "G_X", directory, epoch)
        self._save_network(self.F_Y, "F_Y", directory, epoch)
        self._save_network(self.D_X, "D_X", directory, epoch)
        self._save_network(self.D_Y, "D_Y", directory, epoch)

    def _save_network(self, network: torch.nn.Module, name, directory, epoch):
        filename = "%s/net_%s.pth" % (epoch, name)
        path = os.path.join(directory, filename)

        torch.save(network.state_dict(), path)
