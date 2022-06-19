from enum import Enum
from torch import nn
import torch


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
        return nn.BatchNorm2d(num_features, affine=True, track_running_stats=True)
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
            m.bias.data.fill_(0)


class ConvLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernels, stride, padding=0, use_bias=False, inplaced=False, sloped=0.2, activation=Activation.relu, normalization=Normalization.batch, init_method=WeightInit.normal, init_gain=0.02):
        super(ConvLayer, self).__init__()

        self.layers = [
            nn.Conv2d(in_ch, out_ch, kernels, stride, padding, bias=use_bias)]

        if normalization != Normalization.none:
            self.layers += [getNormalizationLayer(normalization, out_ch)]

        if activation != Activation.none:
            self.layers += [getActivationLayer(activation, inplaced, sloped)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class DeconvLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernels, stride, padding=0, output_padding=0, use_bias=False, inplaced=False, sloped=0.2, activation=Activation.relu, normalization=Normalization.batch, init_method=WeightInit.normal, init_gain=0.02) -> None:
        super(DeconvLayer, self).__init__()

        self.layers = [nn.ConvTranspose2d(
            in_ch, out_ch, kernels, stride, padding, output_padding=output_padding, bias=use_bias
        )]

        if normalization != Normalization.none:
            self.layers += [getNormalizationLayer(normalization, out_ch)]

        if activation != Activation.none:
            self.layers += [getActivationLayer(activation, inplaced, sloped)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class ResidualLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernels, stride, normalization=Normalization.batch, padding_type=Padding.reflect, init_method=WeightInit.normal, init_gain=0.02, use_bias=False, inplace=False):
        super(ResidualLayer, self).__init__()

        self.layers = []

        padw = 1

        if padding_type != Padding.none and padding_type != Padding.zero:
            self.layers += [getPaddingLayer(padding_type, padw)]

        self.layers += [ConvLayer(in_ch, out_ch, kernels, stride, padw if padding_type ==
                                  Padding.zero else 0, normalization=normalization, use_bias=use_bias, inplaced=inplace)]

        if padding_type != Padding.none and padding_type != Padding.zero:
            self.layers += [getPaddingLayer(padding_type, padw)]

        self.layers += [ConvLayer(in_ch, out_ch, kernels, stride, padw if padding_type ==
                                  Padding.zero else 0, normalization=normalization, use_bias=use_bias, inplaced=inplace)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x) + x


class DownConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, init_method=WeightInit.normal, init_gain=0.02):
        super(DownConvBlock, self).__init__()

        self.layers = [
            # First conv block
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
            # Second conv block
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
        ]

        # Max pooling (downsampling)
        self.downsampling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.model(x)
        return out, self.downsampling(out)


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, init_method=WeightInit.normal, init_gain=0.02):
        super(UpConvBlock, self).__init__()
        self.layers = [
            # First conv block
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
            # Second conv block
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
        ]

        self.deconv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Bilinear 2D upsampling
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x, skip_connection):
        _x = self.upsampling(x)
        up = torch.cat([self.deconv(_x), skip_connection], dim=1)
        out = self.model(up)

        return out


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, outer_module: nn.Module):

        super(UnetSkipConnectionBlock, self).__init__()

        self.layers = [
            DownConvBlock(in_ch, out_ch),
            outer_module,
            UpConvBlock(out_ch * 2, in_ch),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention, self).__init__()
#         self.U = nn.Linear(256, 256)
#         self.W = nn.Linear(256, 256)
#         self.v = nn.Linear(256, 1)
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax(1)

#     def forward(self, img_features, hidden_state):
#         U_h = self.U(hidden_state)
#         W_s = self.W(img_features)
#         att = self.tanh(W_s + U_h)
#         print("Att:", att.shape)

#         alpha = self.softmax(att)
#         context = (img_features * alpha)
#         print("Context:", context.shape)
#         print("Alpha:", alpha.shape)
#         return context, alpha


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's CNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        # Tanh activation function
        self.tanh = nn.Tanh()
        # softmax layer to calculate weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, encoder_dim, width, height)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim, width, height)
        :return: attention weighted encoding, weights
        """

        # (batch_size, width, height, encoder_dim)
        encoder_out = encoder_out.permute(0, 2, 3, 1)
        # (batch_size, width, height, decoder_dim)
        decoder_hidden = decoder_hidden.permute(0, 2, 3, 1)

        # (batch_size, width, height, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # (batch_size, width, height, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # (batch_size, width, height, 1)
        att = self.full_att(self.tanh(att1 * att2))
        # (batch_size, width, height, 1)
        alpha = self.softmax(att)

        # (batch_size, width, height, encoder_dim)
        encoding = (encoder_out * alpha)
        # (batch_size, encoder_dim, width, height)
        encoding = encoding.permute(0, 3, 1, 2)
        # (batch_size, 1, width, height)
        alpha = alpha.permute(0, 3, 1, 2)

        return encoding, alpha
