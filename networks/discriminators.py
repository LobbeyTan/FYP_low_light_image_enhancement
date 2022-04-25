from turtle import forward
from torch import nn

from networks.networks import Activation, ConvLayer, Normalization


class NLayerDiscriminator(nn.Module):

    def __init__(self, in_ch, cf=64, n_layers=3, padw=1, normalization=Normalization.batch) -> None:
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = padw

        # Input layers
        self.layers = [ConvLayer(in_ch, cf, kernels=kw, stride=2, padding=padw,
                                 activation=Activation.leaky_relu, normalization=Normalization.none,)]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            self.layers += [ConvLayer(cf * nf_mult_prev, cf * nf_mult, kernels=kw, stride=2,
                                      padding=padw, normalization=normalization, activation=Activation.leaky_relu)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        self.layers += [ConvLayer(cf * nf_mult_prev, cf * nf_mult, kernels=kw, stride=1,
                                  padding=padw, normalization=normalization, activation=Activation.leaky_relu)]

        # Output layers
        self.layers += [ConvLayer(cf * nf_mult, 1, kernels=kw, stride=1, padding=padw,
                                  normalization=Normalization.none, activation=Activation.none)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
