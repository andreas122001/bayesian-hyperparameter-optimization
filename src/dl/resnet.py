import torch

from torch import nn
from typing import Union


class BasicBlock(nn.Module):
    """
    Implements the basic residual block of the ResNet architecture. Consists of two convolutional layers and a
    residual connection.

    :param ch_in: how many input channels
    :param ch_out: how many output channels
    :param stride: the stride to use for the first layer
    :param kernel_size: the kernel size for both layers
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        kernel_size: Union[int, tuple[int, int]] = 3,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            # First layer
            nn.Conv2d(ch_in, ch_out, kernel_size, padding=1, stride=stride),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
            # Second layer
            nn.Conv2d(ch_out, ch_out, kernel_size, padding=1, stride=1),
            nn.BatchNorm2d(ch_out),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, stride=stride, kernel_size=1),
            nn.BatchNorm2d(ch_out),
        )
        self.activation_out = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layers(x)  # Conv layers
        x = x + self.downsample(residual)  # residual
        x = self.activation_out(x)
        return x

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()


class CustomResNet(nn.Module):

    def __init__(self, channels_in=1, n_classes=10):
        super(CustomResNet, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layers = nn.Sequential(
            self._make_layer(32, 32, n_blocks=1, stride=1, kernel_size=3),
            self._make_layer(32, 64, n_blocks=1, stride=2, kernel_size=3),
            self._make_layer(64, 128, n_blocks=1, stride=2, kernel_size=3),
            self._make_layer(128, 128, n_blocks=1, stride=2, kernel_size=3),
            self._make_layer(128, 128, n_blocks=1, stride=2, kernel_size=3),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)

        x = self.layers(x)

        x = x.flatten(start_dim=1)  # ignore batch dim
        x = self.fc(x)

        return x

    def reset(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def _make_layer(
        self,
        channels_in: int,
        channels_out: int,
        n_blocks: int,
        stride: int = 1,
        kernel_size: Union[int, tuple[int, int]] = 3,
    ) -> nn.Module:
        layers = []
        layers.append(
            BasicBlock(
                channels_in, channels_out, stride=stride, kernel_size=kernel_size
            )
        )
        for _ in range(1, n_blocks):
            layers.append(
                BasicBlock(
                    channels_out, channels_out, stride=1, kernel_size=kernel_size
                )
            )

        return nn.Sequential(*layers)


if __name__ == "__main__":

    from time import perf_counter
    from tqdm import tqdm

    t0 = perf_counter()
    resnet = CustomResNet(channels_in=1, n_classes=10)
    for i in tqdm(range(625)):
        t = torch.rand(96, 1, 28, 28)
        o = resnet(t)
        print(o)
    t1 = perf_counter()
    print(f"{t1 - t0:.2f} seconds")
