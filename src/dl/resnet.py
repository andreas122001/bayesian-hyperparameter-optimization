import torch

from torch import nn
from typing import Union


class BasicBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        kernel_size: Union[int, tuple[int, int]] = 3,
    ) -> None:
        super().__init__()
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

    def forward(self, x) -> torch.Tensor:
        residual = x
        x = self.layers(x)  # Conv layers
        x = x + self.downsample(residual)  # residual
        x = self.activation_out(x)
        return x


if __name__ == "__main__":
    t = torch.rand(1, 128, 224, 224)

    block = BasicBlock(128, 64, stride=2)

    o = block(t)
    print(o.shape)
