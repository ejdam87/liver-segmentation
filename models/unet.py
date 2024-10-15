import torch

# Segmentation model

class UNet(torch.nn.Sequential):
    """UNet implementation.

    Implementation based on: https://github.com/milesial/Pytorch-UNet (We modified it for our purposes)
    """

    def __init__(self, in_channels: int, out_channels: int, out_activation: torch.nn.Module) -> None:
        """
        in_channels : # How many channels does the input image have (1 - grayscale, 3 - rgb)
        out_channels : How many channels does the output image have (same meaning as <in_channels>)
        """
        super().__init__()

        self.input_block = UNetBaseBlock(in_channels, 64)
        self.down_block_1 = UNetDownBlock(64, 128)
        self.down_block_2 = UNetDownBlock(128, 256)
        self.down_block_3 = UNetDownBlock(256, 512)
        self.down_block_4 = UNetDownBlock(512, 1024)
        self.up_block_1 = UNetUpBlock(1024, 512)
        self.up_block_2 = UNetUpBlock(512, 256)
        self.up_block_3 = UNetUpBlock(256, 128)
        self.up_block_4 = UNetUpBlock(128, 64)
        self.output_block = torch.nn.Conv2d(64, out_channels, kernel_size=1)

        self.out_activation = out_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates U-Net
        """
        x1 = self.input_block(x)
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x5 = self.down_block_4(x4)
        x = self.up_block_1(x5, [x4])
        x = self.up_block_2(x, [x3])
        x = self.up_block_3(x, [x2])
        x = self.up_block_4(x, [x1])
        x = self.output_block(x)

        return self.out_activation(x)


class UNetBaseBlock(torch.nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int]
    stride: int | tuple[int]
    padding: str | int | tuple[int]

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()

        self.conv_1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batchnorm_1 = torch.nn.BatchNorm2d(out_channels)
        self.batchnorm_2 = torch.nn.BatchNorm2d(out_channels)
        self.relu_1 = torch.nn.ReLU(inplace=True)
        self.relu_2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)
        return x


class UNetDownBlock(UNetBaseBlock):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int]
    stride: int | tuple[int]
    padding: str | int | tuple[int]
    pooling: int | tuple[int]

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.maxpool = torch.nn.MaxPool2d(pooling)

    def forward(self, x):
        x = self.maxpool(x)
        x = super().forward(x)
        return x


class UNetUpBlock(UNetBaseBlock):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int]
    stride: int | tuple[int]
    num_skips: int
    upsample_scale_factor: int

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        num_skips=1
    ):
        super().__init__(in_channels // 2 * (num_skips + 1), out_channels)

        self.upscale = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x, skip):
        x = self.upscale(x)
        delta_width = skip[0].size()[2] - x.size()[2]
        delta_height = skip[0].size()[3] - x.size()[3]
        x = torch.nn.functional.pad(
            x,
            [
                delta_height // 2,
                delta_height - (delta_height // 2),
                delta_width // 2,
                delta_width - (delta_width // 2),
            ],
        )
        x = torch.cat([*skip, x], dim=1)
        x = super().forward(x)
        return x
