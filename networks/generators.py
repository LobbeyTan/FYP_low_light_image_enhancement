from turtle import forward
import torch.nn as nn

from networks.networks import ConvLayer, DeconvLayer, Normalization, ResidualLayer


class ResnetGenerator(nn.Module):

    def __init__(self, in_ch, out_ch, cf=64, n_downsampling=2, n_blocks=6, normalization=Normalization.batch) -> None:
        super(ResnetGenerator, self).__init__()

        self.layers = []

        # Input Layers
        self.layers += [
            
            ConvLayer(in_ch, cf, 9, 1, normalization=normalization),
        ]

        # Downsampling layers
        self.layers += [
            ConvLayer(cf * (2 ** i), cf * (2 ** (i + 1)), 3, 2, normalization=normalization) for i in range(n_downsampling)
        ]

        # Resnet layers
        in_out_ch = cf * (n_downsampling ** 2)
        self.layers += [
            ResidualLayer(in_out_ch, in_out_ch, 3, 1, normalization=normalization) for _ in range(n_blocks)
        ]

        # Upsampling layers
        self.layers += [
            DeconvLayer(cf * (2 ** i), cf * (2 ** (i - 1)), 3, 2, normalization=normalization) for i in range(n_downsampling, 0, -1)
        ]

        # Output layers
        self.layers += [
            ConvLayer(cf, out_ch, 9, 1, normalization=normalization)
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
