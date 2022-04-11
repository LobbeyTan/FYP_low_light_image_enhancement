from enum import Enum
from torch import nn


class Activation(Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    linear = "linear"
    tanh = "tanh"
    none = "none"


class Normalization(Enum):
    batch = "batch"
    instance = "instance"
    none = "none"


class Padding(Enum):
    reflect = "reflect"
    replicate = "replicate"
    zero = "zero"
    none = "none"


class WeightInit(Enum):
    normal = "normal"
    xavier = "xavier"
    kaiming = "kaiming"
    orthogonal = "orthogonal"
    none = "none"


def getActivationLayer(activation: Activation, inplace=False, negative_slope=0.2):
    if activation == Activation.relu:
        return nn.ReLU(inplace)
    elif activation == Activation.leaky_relu:
        return nn.LeakyReLU(negative_slope, inplace)
    elif activation == Activation.tanh:
        return nn.Tanh()
    elif activation == Activation.none:
        return None
    else:
        raise NotImplementedError("Activation layer undefined")


def getNormalizationLayer(normalization: Normalization, num_features):
    if normalization == Normalization.batch:
        return nn.BatchNorm2d(num_features)
    elif normalization == Normalization.instance:
        return nn.InstanceNorm2d(num_features)
    elif normalization == Normalization.none:
        return None
    else:
        raise NotImplementedError("Normalization layer undefined")


def getPaddingLayer(padding_type: Padding, size):
    if padding_type == Padding.reflect:
        return nn.ReflectionPad2d(size)
    elif padding_type == Padding.replicate:
        return nn.ReplicationPad2d(size)
    return None


def initNetWeight(m: nn.Module, method: WeightInit, init_gain=0.02):
    name = m.__class__.__name__

    if method != WeightInit.none and hasattr(m, 'weight'):
        if name.find('Conv') != -1 or name.find('Linear') != -1:
            if method == WeightInit.normal:
                nn.init.normal_(m.weight.data, 0, init_gain)
            elif method == WeightInit.xavier:
                nn.init.xavier_normal_(m.weight.data, init_gain)
            elif method == WeightInit.kaiming:
                nn.init.kaiming_normal_(m.weight.data)
            elif method == WeightInit.orthogonal:
                nn.init.orthogonal_(m.weight.data, init_gain)
            else:
                raise NotImplementedError("Initialization method undefined")
        elif name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1, init_gain)
            nn.init.constant_(m.bias.data, 0.0)


class ConvLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernels, stride, padding=0, inplaced=False, sloped=0.2, activation=Activation.relu, normalization=Normalization.batch, init_method=WeightInit.normal, init_gain=0.02):
        super(ConvLayer, self).__init__()

        self.layers = [nn.Conv2d(in_ch, out_ch, kernels, stride, padding)]

        if normalization != Normalization.none:
            self.layers += [getNormalizationLayer(normalization, out_ch)]

        if activation != Activation.none:
            self.layers += [getActivationLayer(activation, inplaced, sloped)]

        self.model = nn.Sequential(*self.layers)

        self.model.apply(lambda m: initNetWeight(m, init_method, init_gain))

    def forward(self, x):
        return self.model(x)


class DeconvLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernels, stride, padding=0, output_padding=0, inplaced=False, sloped=0.2, activation=Activation.relu, normalization=Normalization.batch, init_method=WeightInit.normal, init_gain=0.02) -> None:
        super(DeconvLayer, self).__init__()

        self.layers = [nn.ConvTranspose2d(
            in_ch, out_ch, kernels, stride, padding, output_padding=output_padding
        )]

        if normalization != Normalization.none:
            self.layers += [getNormalizationLayer(normalization, out_ch)]

        if activation != Activation.none:
            self.layers += [getActivationLayer(activation, inplaced, sloped)]

        self.model = nn.Sequential(*self.layers)

        self.model.apply(lambda m: initNetWeight(m, init_method, init_gain))

    def forward(self, x):
        return self.model(x)


class ResidualLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernels, stride, normalization=Normalization.batch, padding_type=Padding.reflect, init_method=WeightInit.normal, init_gain=0.02):
        super(ResidualLayer, self).__init__()

        self.layers = []

        padw = 1

        if padding_type != Padding.none and padding_type != Padding.zero:
            self.layers += [getPaddingLayer(padding_type, padw)]

        self.layers += [ConvLayer(in_ch, out_ch, kernels, stride, padw if padding_type ==
                                  Padding.zero else 0, normalization=normalization)]

        if padding_type != Padding.none and padding_type != Padding.zero:
            self.layers += [getPaddingLayer(padding_type, padw)]

        self.layers += [ConvLayer(in_ch, out_ch, kernels, stride, padw if padding_type ==
                                  Padding.zero else 0, normalization=normalization)]

        self.model = nn.Sequential(*self.layers)

        self.model.apply(lambda m: initNetWeight(m, init_method, init_gain))

    def forward(self, x):
        return self.model(x) + x
