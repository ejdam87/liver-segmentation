import torch

class BaselineCNN(torch.nn.Module):
    """
    Baseline model
    """

    def __init__(self, in_channels: int, out_channels: int, out_act: torch.nn.Module) -> None:
        """
        in_channels : How many channels does the input image have (1 - grayscale, 3 - rgb)
        out_channels : How many channels does the output image have (same meaning as <in_channels>)
        """
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding="same")
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_activation = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv(x)))
        x = self.upsample(x)
        x = self.output(x)
        return self.out_activation(x)
