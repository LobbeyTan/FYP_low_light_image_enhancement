import os
from random import randint
import torch
from torch import Tensor, nn
from torchvision import transforms
from models.vgg import load_vgg16
from networks.discriminators import NLayerDiscriminator
from networks.generators import UnetGenerator
from networks.loss_functions import GANLoss, SelfFeaturePreservingLoss
from networks.networks import Normalization


class EnlightenGAN(nn.Module):

    def __init__(self, use_ragan=True, n_patch=5, patch_size=32, lr=0.001, beta1=0.5, device=torch.device("cpu")) -> None:
        super(EnlightenGAN, self).__init__()

        self.lr = lr
        self.device = device
        self.n_patch = n_patch
        self.use_ragan = use_ragan
        self.patch_size = patch_size

        self.vgg = load_vgg16("./models/pretrained/vgg16.weight", self.device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.G = UnetGenerator().to(self.device)

        self.D = NLayerDiscriminator(
            3, n_layers=5, normalization=Normalization.none,
        ).to(self.device)

        self.patch_D = NLayerDiscriminator(
            3, n_layers=4, padw=2, normalization=Normalization.none,
        ).to(self.device)

        self.criterionGAN = GANLoss().to(self.device)
        self.SFP_loss = SelfFeaturePreservingLoss().to(self.device)
        self.SFP_patch_loss = SelfFeaturePreservingLoss().to(self.device)

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=self.lr, betas=(beta1, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.lr, betas=(beta1, 0.999)
        )

        self.optimizer_patch_D = torch.optim.Adam(
            self.patch_D.parameters(), lr=self.lr, betas=(beta1, 0.999)
        )

    def set_input(self, batch_data):
        self.input_A = batch_data['img_A'].to(self.device)
        self.input_A_gray = batch_data['gray_A'].to(self.device)
        self.input_B = batch_data['img_B'].to(self.device)
        self.image_paths = batch_data['path_A']

    def forward(self):

        self.fake_B, self.latent_real_A = self.G(
            self.input_A, self.input_A_gray)

        h = self.input_A.size(2)
        w = self.input_A.size(3)

        self.fake_patch = []
        self.target_patch = []
        self.source_patch = []

        for _ in range(self.n_patch):
            w_offset = randint(0, max(0, w - self.patch_size - 1))
            h_offset = randint(0, max(0, h - self.patch_size - 1))

            self.fake_patch.append(
                self.fake_B[:, :, h_offset: (h_offset+self.patch_size),
                            w_offset: (w_offset + self.patch_size)]
            )

            self.target_patch.append(
                self.input_B[:, :, h_offset: (h_offset+self.patch_size),
                             w_offset: (w_offset + self.patch_size)]
            )

            self.source_patch.append(
                self.input_A[:, :, h_offset: (h_offset+self.patch_size),
                             w_offset: (w_offset + self.patch_size)]
            )

    def _backward_D(self, D: nn.Module, real: torch.Tensor, fake: torch.Tensor, use_ragan=True):
        pred_real = D(real)
        pred_fake = D(fake.detach())

        if self.use_ragan and use_ragan:
            loss_D_real = self.criterionGAN(
                pred_real - torch.mean(pred_fake), is_real=True
            )
            loss_D_fake = self.criterionGAN(
                pred_fake - torch.mean(pred_real), is_real=False
            )
        else:
            loss_D_real = self.criterionGAN(pred_real, is_real=True)
            loss_D_fake = self.criterionGAN(pred_fake, is_real=False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return loss_D

    def backward_D(self):
        self.loss_D = self._backward_D(
            self.D, self.input_B, self.fake_B, use_ragan=True
        )

        self.loss_D.backward()

    def backward_patch_D(self):
        self.loss_patch_D = 0

        for i in range(self.n_patch):
            self.loss_patch_D += self._backward_D(
                self.patch_D, self.target_patch[i], self.fake_patch[i], True
            )

        self.loss_patch_D /= float(self.n_patch)

        self.loss_patch_D.backward()

    def backward_G(self):
        if self.use_ragan:
            pred_real = self.D(self.input_B)
            pred_fake = self.D(self.fake_B)

            self.loss_G = (self.criterionGAN((pred_real - torch.mean(pred_fake)), is_real=False) +
                           self.criterionGAN((pred_fake - torch.mean(pred_real)), is_real=True)) / 2
        else:
            self.loss_G = self.criterionGAN(self.D(self.fake_B), True)

        self.loss_G_SFP = self.SFP_loss(self.vgg, self.fake_B, self.input_A)

        self.loss_G_patch = 0

        self.loss_G_SFP_patch = 0

        for i in range(self.n_patch):
            if self.use_ragan:
                pred_real_patch = self.patch_D(self.target_patch[i])
                pred_fake_patch = self.patch_D(self.fake_patch[i])

                self.loss_G_patch += (self.criterionGAN((pred_real_patch - torch.mean(pred_fake_patch)), is_real=False) +
                                      self.criterionGAN((pred_fake_patch - torch.mean(pred_real_patch)), is_real=True)) / 2
            else:
                self.loss_G_patch += self.criterionGAN(
                    self.D(self.fake_patch[i]), is_real=True
                )

            self.loss_G_SFP_patch += self.SFP_patch_loss(
                self.vgg, self.fake_patch[i], self.source_patch[i]
            )

        self.loss_G_patch /= float(self.n_patch)
        self.loss_G_SFP_patch /= float(self.n_patch)

        self.total_loss_G = self.loss_G_SFP + \
            self.loss_G_SFP_patch + self.loss_G + self.loss_G_patch

        self.total_loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Update G
        self.D.requires_grad_(False)
        self.patch_D.requires_grad_(False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Update Global D
        self.D.requires_grad_(True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Update Local D
        self.patch_D.requires_grad_(True)
        self.optimizer_patch_D.zero_grad()
        self.backward_patch_D()
        self.optimizer_patch_D.step()

    def save_model(self, directory, epoch):
        self._save_network(self.G, "G", directory, epoch)
        self._save_network(self.D, "D", directory, epoch)
        self._save_network(self.patch_D, "patch_D", directory, epoch)

    def _save_network(self, network: torch.nn.Module, name, directory, epoch):
        directory = os.path.join(directory, "iter_%s/" % epoch)

        try:
            os.makedirs(directory)
        except:
            pass

        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        torch.save(network.state_dict(), path)

    def load_model(self, directory):
        self._load_model(self.G, "G", directory)
        self._load_model(self.D, "D", directory)
        self._load_model(self.patch_D, "patch_D", directory)

    def _load_model(self, network: torch.nn.Module, name, directory):
        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        network.load_state_dict(torch.load(path, map_location=self.device))

        network.to(self.device)

    def eval(self):
        self.G.eval()
        self.D.eval()
        self.patch_D.eval()

    def test(self, save_dir: str):
        with torch.no_grad():
            self.forward()
            return self.save_result(save_dir)

    def save_result(self, save_dir):
        paths = []
        inverse_transform = transforms.Compose([transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
                                                transforms.Normalize(
                                                    [-0.5, -0.5, -0.5], [1., 1., 1.]),
                                                transforms.ToPILImage()
                                                ])

        def save(result: Tensor, name: str, result_dir: str):
            img = inverse_transform(result)
            path = os.path.join(result_dir, f"{name}.jpg")
            img.save(path, "JPEG")
            paths.append((path, name))

        batch_size = self.real_X.shape[0]

        for i in range(batch_size):

            result_dir = os.path.join(save_dir, f"output_{i+1}")

            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)

            save(self.input_A[i], "Real X", result_dir)
            save(self.input_B[i], "Real Y", result_dir)
            save(self.fake_B[i], "Fake Y", result_dir)

        return paths
