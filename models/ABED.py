import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        padding_mode="replicate",
        bias=bias,
        groups=groups,
    )


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="trilinear", scale_factor=(1, 2, 2)),
            conv1x1(in_channels, out_channels),
        )


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def batch_norm(in_channels):
    return nn.BatchNorm3d(in_channels)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU/SiLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, activation=nn.ReLU()):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.norm = batch_norm(self.out_channels)
        self.activation = activation

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.activation(self.norm(self.conv1(x)))
        x = self.activation(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x  # , before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU/SiLU activation follows each convolution.
    """

    def __init__(
        self, in_channels, out_channels, up_mode="transpose", activation=nn.ReLU()
    ):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.norm = batch_norm(out_channels)
        self.activation = activation

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

    def forward(self, x):
        x = self.upconv(x)

        x = self.activation(self.norm(x))
        # x = self.activation(self.conv2(x))
        return x


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        Attention modules                              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class SEA(nn.Module):
    """
    3-D Sequence attention module SEA :
    performs 3-D average pooling to contract spatial info and keep temporal, two conv 3D and sigmoid
    join a weight to each time step in order to capture the most important time related features
    """

    def __init__(self, channels, lat_size, lon_size):
        super(SEA, self).__init__()

        self.channels = channels
        self.kernel_size = (1, lat_size, lon_size)

        self.average_pool = nn.AvgPool3d(
            kernel_size=self.kernel_size
        )  # 1,ikmage size ?
        self.conv1 = conv1x1(self.channels, 2 * self.channels)
        self.conv2 = conv1x1(2 * self.channels, self.channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.average_pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return out * x


class SPA(nn.Module):
    """
    3-D Spatial attention module SPA :
    performs 3-D conv to contract time related information and sigmoid to join a
    weight to each pixel to capture the most important spatial features
    """

    def __init__(self, channels, time_size):
        super(SPA, self).__init__()
        self.channels = channels
        self.kernel = (time_size, 1, 1)
        self.conv = nn.Conv3d(self.channels, self.channels, self.kernel)  # 10,1,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.conv(x))

        return out * x


class SSAB(nn.Module):
    """
    Sequence and Spatial Attention Block
    """

    def __init__(self, channels, input_size):
        super(SSAB, self).__init__()
        self.channels = channels
        self.time_size = input_size[0]

        # if input_size[1]!=input_size[2]:
        #     raise ValueError(f"image size is not squared anymore {input_size}")

        # self.image_size=input_size[1]
        self.lat_size = input_size[1]
        self.lon_size = input_size[2]
        self.conv1 = conv3x3(self.channels, self.channels)
        self.conv2 = conv3x3(self.channels, self.channels)
        self.sea = SEA(self.channels, self.lat_size, self.lon_size)
        self.spa = SPA(self.channels, self.time_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sea(out)
        out = self.spa(out)

        return out + x


class RSSAB(nn.Module):
    """
    performs M-SSAB
    the original paper doesnt mention any number so it's basically a parameter to tune
    """

    def __init__(self, n_ssab, channels, input_size):
        super(RSSAB, self).__init__()
        self.input_size = input_size
        self.n_ssab = n_ssab
        self.channels = channels

        self.ssabs = []
        for i in range(self.n_ssab):
            ssab = SSAB(self.channels, self.input_size)
            self.ssabs.append(ssab)

        self.conv = conv3x3(self.channels, self.channels)
        self.ssabs = nn.ModuleList(self.ssabs)

    def forward(self, x):
        for i, module in enumerate(self.ssabs):
            out = module(x) if i == 0 else module(out)

        out = self.conv(out)

        return out + x


class ABED(nn.Module):
    """`
    Attention Based Encoder Decoder(ABED)` class is inspired by the article : ED-DRAP: Encoder–Decoder Deep Residual Attention Prediction Network for Radar Echoes
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9674896&isnumber=9651998
    and the CBAM attention layer : "CBAM: Convolutional Block Attention Module"

    The ABED network is a convolutional encoder-decoder neural network
    with deep attention modules at different resolutions.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Deep spatial and temporal attentions modules are added
    in the decoding pathway to gather the essential information.


    """

    def __init__(
        self,
        size_input=(288, 160, 184),
        activation="ReLU",
        channel_input=2,
        channel_output=2,
        filters=[16, 32, 64],
    ):
        super(ABED, self).__init__()

        if activation in ("ReLU", "SiLU"):
            if activation == "ReLU":
                self.activation = nn.ReLU()
            else:
                self.activation = nn.SiLU()
        else:
            raise ValueError(
                '"{}" is not a valid mode for '
                'activation. Only "SiLU" and '
                '"ReLU" are allowed.'.format(activation)
            )
        if channel_output > channel_input:
            raise ValueError(
                "can't add residul frame on different numbers of channels"
                "please check that number of inputs channels is greater than or equal to"
                "the number of output channels"
            )

        self.num_classes = channel_output  # attributes needed in the the model class
        self.nb_inputs = channel_input  # same idea, all the code needs to be cleaned
        self.channel_input = channel_input
        self.channel_output = channel_output
        self.filters = filters
        self.size_input = size_input

        # self.USE_MASK="MASK" in self.data

        # list module :
        # Down way
        self.norm = batch_norm(self.channel_input)
        self.first_conv = conv3x3(self.channel_input, self.filters[0])
        self.Down_conv1 = DownConv(self.filters[0], self.filters[1])
        self.Down_conv2 = DownConv(self.filters[1], self.filters[2])

        # upway
        # self.Up_conv1=UpConv(self.filters[2], self.filters[1])
        # self.Up_conv2=UpConv(self.filters[1], self.filters[0])
        self.Up_conv1 = UpConv(self.filters[2], self.filters[1], up_mode="trilinear")
        self.Up_conv2 = UpConv(self.filters[1], self.filters[0], up_mode="trilinear")
        self.final_conv = conv3x3(self.filters[0], self.channel_output)

        # Attention

        self.attention1 = RSSAB(
            4,
            self.filters[2],
            (self.size_input[0], self.size_input[1] // 4, self.size_input[2] // 4),
        )
        self.attention2 = RSSAB(
            2,
            self.filters[1],
            (self.size_input[0], self.size_input[1] // 2, self.size_input[2] // 2),
        )
        self.attention3 = RSSAB(1, self.filters[0], self.size_input)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.norm(x)

        x = self.first_conv(x)

        x = self.Down_conv1(x)

        x = self.Down_conv2(x)

        x = self.attention1(x)

        x = self.Up_conv1(x)

        x = self.attention2(x)

        x = self.Up_conv2(x)

        x = self.attention3(x)

        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    # example
    model = ABED(
        size_input=(72, 160, 184),
        activation="ReLU",
        channel_input=11,
        channel_output=2,
        filters=[4, 8, 16],
    )
    # print(model)
    x = torch.randn((8, 11, 72, 160, 184))
    y = torch.randn((8, 2, 72, 160, 184))

    yhat = model(x)
    print(yhat.shape)
