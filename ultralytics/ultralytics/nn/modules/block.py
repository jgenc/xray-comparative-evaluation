# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "ResNet101_1",
    "ResNet101_2",
    "ResNet101_3",
    "MobileNet1",
    "MobileNet2",
    "MobileNet3",
    "EfficientNetV2_1",
    "EfficientNetV2_2",
    "EfficientNetV2_3",
    "DOAM",
    "UpsampleAdd",
    "LIM_DFPN",
    "ManyToOne",
    "DDoM",
    "DDoMClf",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(
            c_, c_, 2, 2, 0, bias=True
        )  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()
    ):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        print(f"[\x1b[1;31mDEBUG\x1b[0m] SPPF: c1: {c1}, c2: {c2}, c_: {c_}")
        self.debug_shapes = (c1, c2, c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        # print(f"[\x1b[1;31mDEBUG\x1b[0m] SPPF Input size: {x.size()}")
        # print(f"[\x1b[1;31mDEBUG\x1b[0m] SPPF Debug Shapes: {self.debug_shapes}")
        # print(f"[\x1b[1;31mDEBUG\x1b[0m] SPPF before cv1")
        x = self.cv1(x)
        # print(f"[\x1b[1;31mDEBUG\x1b[0m] SPPF after cv1")
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(
            *(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(
                Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1)
                for _ in range(n)
            )
        )


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = (
            nn.Sequential(Conv(c1, c3, k=1, s=s, act=False))
            if s != 1 or c1 != c3
            else nn.Identity()
        )

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


"""

Our code modifications used in the `nn.modules.block` module, in order to support Next-ViT, LIM, DOAM, and CHR.

"""


class MobileNet1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileNet2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        modules = list(model.children())
        modules = modules[0][4:9]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileNet3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        modules = list(model.children())
        modules = modules[0][9:]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


"""
    ResNet 101.
"""


class ResNet101_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.model = nn.Sequential(
            *list(model.children())[:-4]
        )  # until layer1 inclusive

    def forward(self, x):
        return self.model(x)


class ResNet101_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[-4])  # only layer2

    def forward(self, x):
        return self.model(x)


class ResNet101_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[-3])  # layer3 and layer4

    def forward(self, x):
        return self.model(x)


"""
    EfficientNetV2
"""


class EfficientNetV2_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_v2_s()
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class EfficientNetV2_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_v2_s()
        modules = list(model.children())
        modules = modules[0][4:6]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class EfficientNetV2_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_v2_s()
        modules = list(model.children())
        modules = modules[0][6:]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


"""
    Auxiliary modules.
"""
"""
    Basic HR module: Upsample and Concatenation.
"""


class UpsampleAdd(nn.Module):
    def __init__(self, dimension=1):
        super(UpsampleAdd, self).__init__()
        self.d = dimension

    def forward(self, ip):
        x = ip[0]
        y = ip[1]

        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W))
        out = torch.cat([z, y], self.d)
        return out


"""
    LIM
"""


class Boundary_Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Boundary_Aggregation, self).__init__()
        self.conv = nn.Conv2d(in_channels * 5, in_channels, 1)

    def forward(self, x_batch: torch.tensor):
        in_channels, height, width = x_batch.size()[1:4]
        x_clk_rot = torch.rot90(x_batch, -1, [2, 3])
        x1 = self.up_to_bottom(x_batch, in_channels, height)
        x2 = self.bottom_to_up(x_batch, in_channels, height)
        x3 = self.left_to_right(x_clk_rot, in_channels, width)
        x4 = self.right_to_left(x_clk_rot, in_channels, width)
        x_con = torch.cat((x_batch, x1, x2, x3, x4), 1)
        x_merge = self.conv(x_con)
        return x_merge

    def left_to_right(self, x_clk_rot: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_clk_rot)
        x = self.up_to_bottom(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def right_to_left(self, x_clk_rot: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_clk_rot)
        x = self.bottom_to_up(x, in_channels, height)
        x = torch.rot90(x, 1, [2, 3])
        return x

    def bottom_to_up(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height - 1, -1, -1):
            x[:, :, i] = torch.max(x[:, :, i:], 2, True)[0].squeeze(2)
        return x

    def up_to_bottom(self, x_raw: torch.tensor, in_channels: int, height: int):
        x = torch.clone(x_raw)
        for i in range(height):
            x[:, :, i] = torch.max(x[:, :, : i + 1], 2, True)[0].squeeze(2)
        return x


class LIM_DFPN(nn.Module):
    def __init__(self, ignore):
        super(LIM_DFPN, self).__init__()

        C3_size, C4_size, C5_size = 256, 512, 1024
        feature_size = 256

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1_dual = nn.Conv2d(
            C5_size, feature_size, kernel_size=1, stride=1, padding=0
        )
        self.P5_upsampled_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_upsampled_2 = nn.Upsample(scale_factor=2, mode="nearest")
        # self.P5_upsampled_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1_dual = nn.Conv2d(
            C4_size, feature_size, kernel_size=1, stride=1, padding=0
        )
        self.P4_upsampled_1 = nn.Upsample(scale_factor=2, mode="nearest")
        # self.P4_upsampled_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_dual = nn.Conv2d(
            C3_size, feature_size, kernel_size=1, stride=1, padding=0
        )
        #         self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P3 elementwise to C2
        # self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=2, padding=1
        )

        # self.conv_to_1 = nn.Conv2d(feature_size, 16, kernel_size=1)
        # self.conv_resume = nn.Conv2d(16, feature_size, kernel_size=1)

        self.corner_proc_C3 = Boundary_Aggregation(256)
        self.corner_proc_C4 = Boundary_Aggregation(512)
        self.corner_proc_C5 = Boundary_Aggregation(1024)

        self.conv_p3_to_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv_p4_to_p5 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.conv_p3_to_p5 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)

        # self.sig_3 = nn.Sigmoid()
        # self.sig_4 = nn.Sigmoid()
        # self.sig_5 = nn.Sigmoid()
        self.gamma_3 = nn.Parameter(torch.zeros(1))
        self.gamma_4 = nn.Parameter(torch.zeros(1))
        self.gamma_5 = nn.Parameter(torch.zeros(1))

        # self.conv_down_3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_down_4 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        # self.conv_down_5 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        global count
        C3, C4, C5 = inputs  # 150，75,38,19
        # print(C3.size(), C4.size(), C5.size())
        C3_BA = self.corner_proc_C3(C3)
        C4_BA = self.corner_proc_C4(C4)
        C5_BA = self.corner_proc_C5(C5)

        P5_x = self.P5_1(C5)
        P5_upsampled_x_1 = self.P5_upsampled_1(P5_x)  # 38
        P5_upsampled_x_2 = self.P5_upsampled_2(P5_upsampled_x_1)[:, :, :, :]  # 75
        # P5_upsampled_x_3 = self.P5_upsampled_3(P5_upsampled_x_2)#150
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x_1  # 38
        P4_upsampled_x_1 = self.P4_upsampled_1(P4_x)[:, :, :, :]  # 75
        # P4_upsampled_x_2 = self.P4_upsampled_2(P4_upsampled_x_1)#150
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        # print(P3_x.size(), P4_upsampled_x_1.size(), P5_upsampled_x_2.size())

        P3_x = P3_x + P4_upsampled_x_1 + P5_upsampled_x_2  # 75
        # P3_upsampled_x_1 = self.P3_upsampled(P3_x)  # 150
        P3_x = self.P3_2(P3_x)

        P3_dual = self.P3_1_dual(C3_BA)
        P3_downsample_x_1 = self.conv_p3_to_p4(P3_dual)
        P3_downsample_x_2 = self.conv_p3_to_p5(P3_downsample_x_1)

        P4_dual = self.P4_1_dual(C4_BA)
        P4_dual = P4_dual + P3_downsample_x_1
        P4_downsample_x_1 = self.conv_p4_to_p5(P4_dual)

        P5_dual = self.P5_1_dual(C5_BA)
        P5_dual = P5_dual + P4_downsample_x_1 + P3_downsample_x_2

        # P2_x = self.P2_1(C2)
        # P2_x = P2_x + P3_upsampled_x_1 +  P4_upsampled_x_2 + P5_upsampled_x_3# 75
        # P2_x = self.P2_2(P2_x)

        P6_x = self.P6(C5)  # 10

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)  # 5

        # print(P3_dual.shape)
        # img = P3_dual[0][0].cpu().detach().numpy()
        # for i in range(1,P3_dual.shape[1]):
        #    img = img + P3_dual[0][i].cpu().detach().numpy()
        # print('img_shape',img.shape)
        # cv2.imwrite('test/test'+str(count)+'.jpg',img * 255)
        # time.sleep(10)
        O3_x = self.gamma_3 * P3_dual + (1 - self.gamma_3) * P3_x
        O4_x = self.gamma_4 * P4_dual + (1 - self.gamma_4) * P4_x
        O5_x = self.gamma_5 * P5_dual + (1 - self.gamma_5) * P5_x
        return [O3_x, O4_x, O5_x, P6_x, P7_x]


"""
    Takes in a list of tensors and returns a unique tensor according to the index
"""


class ManyToOne(nn.Module):
    def __init__(self, ip_ch, op_ch, idx):
        super(ManyToOne, self).__init__()
        self.idx = idx

    def forward(self, ip):
        op = ip[self.idx].clone()
        return op


"""
    DOAM.
"""


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        batch_norm=True,
        activation=torch.nn.LeakyReLU(0.2, inplace=True),
    ):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class RIA(nn.Module):
    def __init__(self):
        super(RIA, self).__init__()
        self.AdaptiveAverPool_5 = torch.nn.Sequential()
        self.AdaptiveAverPool_5.add_module(
            "AdaptiveAverPool_5", nn.AdaptiveAvgPool2d((60, 60))
        )

        self.AdaptiveAverPool_10 = torch.nn.Sequential()
        self.AdaptiveAverPool_10.add_module(
            "AdaptiveAverPool_10", nn.AdaptiveAvgPool2d((30, 30))
        )

        self.AdaptiveAverPool_15 = torch.nn.Sequential()
        self.AdaptiveAverPool_15.add_module(
            "AdaptiveAverPool_15", nn.AdaptiveAvgPool2d((20, 20))
        )

        self.GatedConv2dWithActivation = GatedConv2dWithActivation(
            in_channels=(8 * 3),
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=None,
        )

        self.conv2d_1_rgb_red_concat_5 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_5.add_module(
            "conv2d_1_rgb_red_concat_5", nn.Conv2d(16, 8, kernel_size=3, padding=1)
        )
        self.conv2d_1_rgb_red_concat_10 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_10.add_module(
            "conv2d_1_rgb_red_concat_10", nn.Conv2d(16, 8, kernel_size=3, padding=1)
        )
        self.conv2d_1_rgb_red_concat_15 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_15.add_module(
            "conv2d_1_rgb_red_concat_15", nn.Conv2d(16, 8, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_pooled_upsample_5 = torch.zeros(
            (x.shape[0], 8, x.shape[2], x.shape[3]), dtype=x.dtype
        ).to(x.device)
        x_pooled_upsample_10 = torch.zeros(
            (x.shape[0], 8, x.shape[2], x.shape[3]), dtype=x.dtype
        ).to(x.device)
        x_pooled_upsample_15 = torch.zeros(
            (x.shape[0], 8, x.shape[2], x.shape[3]), dtype=x.dtype
        ).to(x.device)

        x_pooled_5 = self.AdaptiveAverPool_5(x)
        x_pooled_10 = self.AdaptiveAverPool_10(x)
        x_pooled_15 = self.AdaptiveAverPool_15(x)

        for i in range(60):
            for j in range(60):
                x_pooled_upsample_5[:, :, i * 5 : (i + 1) * 5, j * 5 : (j + 1) * 5] = (
                    x_pooled_5[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                )

        for i in range(30):
            for j in range(30):
                x_pooled_upsample_10[
                    :, :, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10
                ] = x_pooled_10[:, :, i, j].unsqueeze(-1).unsqueeze(-1)

        for i in range(20):
            for j in range(20):
                x_pooled_upsample_15[
                    :, :, i * 15 : (i + 1) * 15, j * 15 : (j + 1) * 15
                ] = x_pooled_15[:, :, i, j].unsqueeze(-1).unsqueeze(-1)

        # x_pooled_upsample_5 = x_pooled_5.repeat_interleave(5, dim=2).repeat_interleave(5, dim=3)
        # x_pooled_upsample_10 = x_pooled_10.repeat_interleave(10, dim=2).repeat_interleave(10, dim=3)
        # x_pooled_upsample_15 = x_pooled_15.repeat_interleave(15, dim=2).repeat_interleave(15, dim=3)

        x_concat_5 = torch.cat([x, x_pooled_upsample_5], 1)
        x_concat_10 = torch.cat([x, x_pooled_upsample_10], 1)
        x_concat_15 = torch.cat([x, x_pooled_upsample_15], 1)

        x_concat_5_out = self.conv2d_1_rgb_red_concat_5(x_concat_5)
        x_concat_10_out = self.conv2d_1_rgb_red_concat_10(x_concat_10)
        x_concat_15_out = self.conv2d_1_rgb_red_concat_15(x_concat_15)

        x_gated_conv_input = torch.cat((x_concat_5_out, x_concat_10_out), 1)
        x_gated_conv_input = torch.cat((x_gated_conv_input, x_concat_15_out), 1)

        x_gated_conv_output = self.GatedConv2dWithActivation(x_gated_conv_input)
        return x_gated_conv_output


class DOAM(nn.Module):
    def __init__(self, ignore):
        super(DOAM, self).__init__()
        torch.use_deterministic_algorithms(False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Sequential()
        self.sigmoid.add_module("Sigmoid", nn.Sigmoid())

        self.conv2d_1_1 = torch.nn.Sequential()
        self.conv2d_1_1.add_module(
            "conv2d_1_1", nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

        # rgb convolution
        self.conv2d_1_rgb_attention = torch.nn.Sequential()
        self.conv2d_1_rgb_attention.add_module(
            "conv2d_1_rgb_attention", nn.Conv2d(3, 8, kernel_size=3, padding=1)
        )
        self.conv2d_2_rgb_attention = torch.nn.Sequential()
        self.conv2d_2_rgb_attention.add_module(
            "conv2d_2_rgb_attention", nn.Conv2d(8, 16, kernel_size=3, padding=1)
        )
        self.conv2d_3_rgb_attention = torch.nn.Sequential()
        self.conv2d_3_rgb_attention.add_module(
            "conv2d_3_rgb_attention", nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        self.conv2d_4_rgb_attention = torch.nn.Sequential()
        self.conv2d_4_rgb_attention.add_module(
            "conv2d_4_rgb_attention", nn.Conv2d(32, 16, kernel_size=3, padding=1)
        )
        self.conv2d_5_rgb_attention = torch.nn.Sequential()
        self.conv2d_5_rgb_attention.add_module(
            "conv2d_5_rgb_attention", nn.Conv2d(16, 8, kernel_size=3, padding=1)
        )

        self.ria = RIA()

    def forward(self, im):
        rgb_red = im

        rgb_conved = self.conv2d_1_rgb_attention(rgb_red)
        rgb_conved = self.conv2d_2_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_3_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_4_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_5_rgb_attention(rgb_conved)

        rgb_conved = self.ria(rgb_conved)

        rgb_red_conved = rgb_conved
        rgb_red_conved = self.conv2d_1_1(rgb_red_conved)

        sigmoid_output = self.sigmoid(rgb_red_conved)

        rgb_red = self.gamma * (sigmoid_output * rgb_red) + (1 - self.gamma) * rgb_red

        # output = self.conv2d(rgb_red)

        # edge_detect = edge_detect.float().cpu().squeeze().detach().numpy()
        # edge_detect = cv2.cvtColor(edge_detect, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(edge_detect)# .convert('L')
        # img.save('edge.jpg')

        # softmax_output = softmax_output*255
        # softmax_output_im = softmax_output.float().cpu().squeeze().detach().numpy()
        # im = Image.fromarray(softmax_output_im).convert('L')
        # im.save('edge_sigmoid_output_im.jpg')
        # rgb_im = og_im*sigmoid_output
        # rgb_im = rgb_im.float().cpu().squeeze().detach().numpy()
        # sigmoid_output = sigmoid_output*255
        # sigmoid_output_im = sigmoid_output.float().cpu().squeeze().detach().numpy()
        # im = Image.fromarray(sigmoid_output_im).convert('L')
        # im.save('edge_sigmoid_output_im.jpg')
        # print(rgb_red.shape, sigmoid_output.shape)

        return rgb_red


"""
    DDoM.
"""

"""
    DDoM Classification head.
    # AVG pooling + FC
"""


class DDoMClf(nn.Module):
    def __init__(self, c1, c2, nc):
        super(DDoMClf, self).__init__()
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.clfL = nn.Linear(c1, nc)
        self.fc = nn.Linear(nc, c2)

    def forward(self, x):
        print(f"[\x1b[1;31mDEBUG\x1b[0m] x size: {x.size()}")
        clc_feat = self.avg_pool(x).view(x.size(0), -1)
        print(f"[\x1b[1;31mDEBUG\x1b[0m] Before clc_out")
        print(f"[\x1b[1;31mDEBUG\x1b[0m] clc_feat size: {clc_feat.size()}")
        clc_out = self.clfL(clc_feat)
        print(f"[\x1b[1;31mDEBUG\x1b[0m] After clc_out and before clc_feat")
        clc_feat = self.fc(clc_out)
        print(f"[\x1b[1;31mDEBUG\x1b[0m] After clc_feat")
        return clc_feat


class DDoM(nn.Module):
    def __init__(self, nc):
        super(DDoM, self).__init__()

        self.clfL = nn.Linear(512, nc)

        self.fc0 = nn.Linear(nc, 256)

        self.sig = nn.Sigmoid()

        self.fc1 = nn.Linear(256, 512)

        self.avg_pool2 = nn.AvgPool2d(8, stride=1)

        self.fc2 = nn.Linear(
            768, 512
        )  # Adjusted to match concatenated size of gs and gs2

        self.avg_pool3 = nn.AvgPool2d(16, stride=1)
        self.fc3 = nn.Linear(
            256 + 512, 256
        )  # Adjusted to match concatenated size of gs and gs3

        self.avg_pool4 = nn.AvgPool2d(32, stride=1)
        self.fc4 = nn.Linear(
            256 + 256, 128
        )  # Adjusted to match concatenated size of gs and gs4

    def forward(self, ip):
        cls_feature, o4, o3, o2, o1 = ip
        bs = o4.size(0)
        # print(o4.size(), o3.size(), o2.size(), o1.size())
        # cls_feature = self.avg_pool2(o4).view(o4.size(0), -1)
        # cls_feature = self.clfL(cls_feature)
        gs = cls_feature  # self.fc0(cls_feature)

        gs1 = torch.bernoulli(self.sig(self.fc1(gs)))
        o4 = o4.contiguous()
        o4 = o4.view(bs, 512, -1)
        gs1 = gs1.unsqueeze(2).expand_as(o4)
        doo4 = o4.mul(gs1).view(bs, 512, 8, 8)

        gs2 = self.avg_pool2(doo4).view(bs, -1)
        gs2 = torch.bernoulli(self.sig(self.fc2(torch.cat([gs, gs2], dim=1))))
        o3 = o3.contiguous()
        o3 = o3.view(bs, 512, -1)
        gs2 = gs2.unsqueeze(2).expand_as(o3)
        doo3 = o3.mul(gs2).view(bs, 512, 16, 16)

        gs3 = self.avg_pool3(doo3).view(bs, -1)
        gs3 = torch.bernoulli(self.sig(self.fc3(torch.cat([gs, gs3], dim=1))))
        o2 = o2.contiguous()
        o2 = o2.view(bs, 256, -1)
        gs3 = gs3.unsqueeze(2).expand_as(o2)
        doo2 = o2.mul(gs3).view(bs, 256, 32, 32)

        gs4 = self.avg_pool4(doo2).view(bs, -1)
        gs4 = torch.bernoulli(self.sig(self.fc4(torch.cat([gs, gs4], dim=1))))
        o1 = o1.contiguous()
        o1 = o1.view(bs, 128, -1)
        gs4 = gs4.unsqueeze(2).expand_as(o1)
        doo1 = o1.mul(gs4).view(bs, 128, 64, 64)
        return doo4, doo3, doo2, doo1
