
from importlib.resources import path
import itertools
import os
from unicodedata import name
import torch
import torchvision.transforms as transforms
from networks.discriminators import NLayerDiscriminator
from networks.generators import ResnetGenerator
from networks.loss_functions import GANLoss
from torch import Tensor, nn

from networks.networks import Normalization


class CycleGANModel:

    def __init__(self, lr=0.001, beta1=0.5, gan_mode="lsgan", lamda_A=10.0, lamda_B=10.0, lambda_idt=0.5, n_blocks=9, normalization=Normalization.instance, device=torch.device("cpu"),) -> None:
        print("Creating cycleGAN on: %s" % device)
        self.device = device

        self.lambda_A = torch.tensor(lamda_A)
        self.lambda_B = torch.tensor(lamda_B)
        self.lambda_idt = torch.tensor(lambda_idt)

        self.G_X = ResnetGenerator(
            3, 3, n_blocks=n_blocks, normalization=normalization).to(self.device)

        self.D_Y = NLayerDiscriminator(3).to(self.device)

        self.F_Y = ResnetGenerator(
            3, 3, n_blocks=n_blocks, normalization=normalization).to(self.device)

        self.D_X = NLayerDiscriminator(3).to(self.device)

        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(
            self.G_X.parameters(), self.F_Y.parameters()), lr=lr, betas=(beta1, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(itertools.chain(
            self.D_X.parameters(), self.D_Y.parameters()), lr=lr, betas=(beta1, 0.999)
        )

    def forward(self, real_X: Tensor, real_Y: Tensor):
        self.real_X = real_X.to(self.device)
        self.real_Y = real_Y.to(self.device)

        self.fake_Y = self.G_X(self.real_X)
        self.fake_X = self.F_Y(self.real_Y)

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

        self.idt_X = self.G_X(self.real_Y)

        self.loss_idt_X = self.criterionIdt(
            self.idt_X, self.real_Y) * self.lambda_B * self.lambda_idt

        self.idt_Y = self.F_Y(self.real_X)

        self.loss_idt_Y = self.criterionIdt(
            self.idt_Y, self.real_X) * self.lambda_A * self.lambda_idt

        self.loss_G_X = self.criterionGAN(self.D_Y(self.fake_Y), is_real=True)
        self.loss_F_Y = self.criterionGAN(self.D_X(self.fake_X), is_real=True)

        self.loss_cycle_X = self.criterionCycle(
            self.reconstruc_X, self.real_X) * self.lambda_A

        self.loss_cycle_Y = self.criterionCycle(
            self.reconstruc_Y, self.real_Y) * self.lambda_B

        self.loss_G = self.loss_G_X + self.loss_F_Y + \
            self.loss_cycle_X + self.loss_cycle_Y + self.loss_idt_X + self.loss_idt_Y

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
        directory = os.path.join(directory, "iter_%d" % epoch)

        try:
            os.mkdir(directory)
        except:
            pass

        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        torch.save(network.state_dict(), path)

    def load_model(self, directory):
        self._load_model(self.G_X, "G_X", directory)
        self._load_model(self.F_Y, "F_Y", directory)
        self._load_model(self.D_X, "D_X", directory)
        self._load_model(self.D_Y, "D_Y", directory)

    def _load_model(self, network: torch.nn.Module, name, directory):
        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        network.load_state_dict(torch.load(path, map_location=self.device))

        network.to(self.device)

    def eval(self):
        self.G_X.eval()
        self.F_Y.eval()
        self.D_X.eval()
        self.D_Y.eval()

    def test(self, real_X: Tensor, real_Y: Tensor, save_dir: str):
        with torch.no_grad():
            self.forward(real_X, real_Y)
            return self.save_result(save_dir)

    def save_result(self, save_dir):
        paths = []
        inverse_transform = transforms.Compose([transforms.Normalize([0., 0., 0.], [1/0.229, 1/0.224, 1/0.225]),
                                                transforms.Normalize(
                                                    [-0.485, -0.456, -0.406], [1., 1., 1.]),
                                                transforms.ToPILImage()
                                                ])

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        def save(result: Tensor, name: str):
            img = inverse_transform(result[0])
            path = os.path.join(save_dir, f"{name}.jpg")
            img.save(path, "JPEG")
            paths.append((path, name))

        save(self.real_X, "Real X")
        save(self.real_Y, "Real Y")
        save(self.fake_X, "Fake X")
        save(self.fake_Y, "Fake Y")
        save(self.reconstruc_X, "Reconstruct X")
        save(self.reconstruc_Y, "Reconstruct Y")

        return paths
