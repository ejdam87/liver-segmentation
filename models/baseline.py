import torch

class BaselineCNN(torch.nn.Module):
    """
    Baseline model
    """

    def __init__(self, in_channels: int, out_channels: int, out_activation: torch.nn.Module) -> None:
        """
        in_channels : How many channels does the input image have (1 - grayscale, 3 - rgb)
        out_channels : How many channels does the output image have (same meaning as <in_channels>)
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.upsample = torch.nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.output = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_activation = out_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.output(x)
        x = self.upsample(x)
        return self.out_activation(x)
