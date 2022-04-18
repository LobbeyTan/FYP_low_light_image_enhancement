from turtle import forward
from numpy import pad
import torch.nn as nn

from networks.networks import Activation, ConvLayer, DeconvLayer, Normalization, Padding, ResidualLayer, getPaddingLayer


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
