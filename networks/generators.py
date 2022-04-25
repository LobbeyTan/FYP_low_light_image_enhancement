import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.networks import Activation, ConvLayer, DeconvLayer, DownConvBlock, Normalization, Padding, ResidualLayer, UnetSkipConnectionBlock, UpConvBlock, getPaddingLayer


class ResnetGenerator(nn.Module):

    def __init__(self, in_ch, out_ch, cf=64, n_downsampling=2, n_blocks=6, normalization=Normalization.batch, padding_type=Padding.reflect) -> None:
        super(ResnetGenerator, self).__init__()

        use_bias = normalization == Normalization.instance

        outer_pad = 3

        self.layers = []

        # Input Layers

        if padding_type != Padding.none and padding_type != Padding.zero:
            self.layers += [getPaddingLayer(padding_type, outer_pad)]

        self.layers += [
            ConvLayer(in_ch, cf, 7, 1, padding=outer_pad if padding_type ==
                      Padding.zero else 0, normalization=normalization, use_bias=use_bias)
        ]

        # Downsampling layers
        self.layers += [
            ConvLayer(cf * (2 ** i), cf * (2 ** (i + 1)), 3, 2, 1, use_bias=use_bias, normalization=normalization) for i in range(n_downsampling)
        ]

        # Resnet layers
        in_out_ch = cf * (2 ** n_downsampling)
        self.layers += [
            ResidualLayer(in_out_ch, in_out_ch, 3, 1, normalization=normalization, padding_type=padding_type, use_bias=use_bias) for _ in range(n_blocks)
        ]

        # Upsampling layers
        self.layers += [
            DeconvLayer(cf * (2 ** i), cf * (2 ** (i - 1)), 3, 2, 1, 1, normalization=normalization, use_bias=use_bias) for i in range(n_downsampling, 0, -1)
        ]

        # Output layers
        if padding_type != Padding.none and padding_type != Padding.zero:
            self.layers += [getPaddingLayer(padding_type, outer_pad)]

        self.layers += [
            ConvLayer(cf, out_ch, 7, 1, padding=outer_pad if padding_type ==
                      Padding.zero else 0, normalization=Normalization.none, activation=Activation.tanh)
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class UnetGenerator(nn.Module):

    def __init__(self) -> None:
        super(UnetGenerator, self).__init__()

        self.downsampling1 = nn.MaxPool2d(2)
        self.downsampling2 = nn.MaxPool2d(2)
        self.downsampling3 = nn.MaxPool2d(2)
        self.downsampling4 = nn.MaxPool2d(2)

        self.down1 = DownConvBlock(4, 32)
        self.down2 = DownConvBlock(32, 64)
        self.down3 = DownConvBlock(64, 128)
        self.down4 = DownConvBlock(128, 256)

        self.conv1 = ConvLayer(256, 512, 3, 1, 1,
                               activation=Activation.leaky_relu)
        self.conv2 = ConvLayer(512, 512, 3, 1, 1,
                               activation=Activation.leaky_relu)

        self.up1 = UpConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.up3 = UpConvBlock(128, 64)
        self.up4 = UpConvBlock(64, 32)

        self.conv3 = ConvLayer(32, 3, 3, 1, 1,
                               activation=Activation.tanh,
                               normalization=Normalization.none,
                               )

    def forward(self, x_in: torch.Tensor, gray: torch.Tensor):
        gray1 = gray
        gray2 = self.downsampling1(gray1)
        gray3 = self.downsampling1(gray2)
        gray4 = self.downsampling1(gray3)
        gray5 = self.downsampling1(gray4)

        x1, x = self.down1(torch.cat([x_in, gray], dim=1))
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x5 = self.conv2(self.conv1(x) * gray5)

        x6 = self.up1(x5, x4 * gray4)
        x7 = self.up2(x6, x3 * gray3)
        x8 = self.up3(x7, x2 * gray2)
        x9 = self.up4(x8, x1 * gray1)

        x10 = self.conv3(x9)

        latent = x10 * gray1

        latent = F.relu(latent)

        x_in = (x_in - torch.min(x_in)) / (torch.max(x_in) - torch.min(x_in))

        output = latent + x_in

        return output, latent
