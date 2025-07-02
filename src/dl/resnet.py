import torch

from torch import nn
from typing import Union

from dataclasses import dataclass


# This could arguably also just be a fixed-size array
@dataclass
class CustomLayerConfig:
    """
    A wrapper class for the layer dimensions for the six layers of the custom ResNet model.
    """

    def __init__(
        self, l0_dim=16, l1_dim=16, l2_dim=32, l3_dim=64, l4_dim=128, l5_dim=256
    ) -> None:
        self.l0 = l0_dim
        self.l1 = l1_dim
        self.l2 = l2_dim
        self.l3 = l3_dim
        self.l4 = l4_dim
        self.l5 = l5_dim


class _BasicBlock(nn.Module):
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
        super(_BasicBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            # First layer
            nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
            # Second layer
            nn.Conv2d(ch_out, ch_out, kernel_size, padding=padding, stride=1),
            nn.BatchNorm2d(ch_out),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, stride=stride, kernel_size=1),
            nn.BatchNorm2d(ch_out),
        )
        self.activation_out = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass in the input.

        :param x: the input to predict on.
        :returns: the model logits.
        """
        residual = x
        x = self.layers(x)  # Conv layers
        x = x + self.downsample(residual)  # residual
        x = self.activation_out(x)
        return x

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the module.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()


class CustomResNet(nn.Module):
    """
    A custom ResNet model for demonstration. Contains six layers, enough to fully downsample the data from the FashionMNIST dataset from (1, 28, 28) to (dim, 1, 1) using strides of two.

    :param channels_in: how many channels the input data has.
    :param n_classes: how many output classes to predict.
    :layer_cfg: the config for the six ResNet layers.
    """

    def __init__(
        self,
        channels_in=1,
        n_classes=10,
        layer_cfg: CustomLayerConfig = CustomLayerConfig(
            l0_dim=16,
            l1_dim=16,
            l2_dim=16,
            l3_dim=16,
            l4_dim=16,
            l5_dim=16,
        ),
    ):
        super(CustomResNet, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(channels_in, layer_cfg.l0, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(layer_cfg.l0),
            nn.ReLU(),
        )
        # Use strides only for downsampling
        self.layers = nn.Sequential(
            self._make_layer(
                layer_cfg.l0, layer_cfg.l1, n_blocks=1, stride=1, kernel_size=3
            ),
            self._make_layer(
                layer_cfg.l1, layer_cfg.l2, n_blocks=1, stride=2, kernel_size=3
            ),
            self._make_layer(
                layer_cfg.l2, layer_cfg.l3, n_blocks=1, stride=2, kernel_size=3
            ),
            self._make_layer(
                layer_cfg.l3, layer_cfg.l4, n_blocks=1, stride=2, kernel_size=3
            ),
            self._make_layer(
                layer_cfg.l4, layer_cfg.l5, n_blocks=1, stride=2, kernel_size=3
            ),
        )
        self.fc = nn.Linear(layer_cfg.l5, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass in the input.

        :param x: the input to predict on.
        :returns: the model logits.
        """
        x = self.conv_init(x)

        x = self.layers(x)

        x = x.flatten(start_dim=1)  # flatten all but batch dim
        x = self.fc(x)

        return x

    def _make_layer(
        self,
        channels_in: int,
        channels_out: int,
        n_blocks: int,
        stride: int = 1,
        kernel_size: Union[int, tuple[int, int]] = 3,
    ) -> nn.Module:
        """
        Creates a layer of basic residual blocks.

        :param channels_in: the input dimensionality.
        :param channels_out: the output dimensionality.
        :param n_block: how many basic blocks to use for the layer.
        :param stride: which stride to use for the first block. Used for downsampling the input.
        :param kernel_size: the kernel size.
        :returns: the PyTorch module containing the layer.
        """
        layers = []
        layers.append(
            _BasicBlock(
                channels_in, channels_out, stride=stride, kernel_size=kernel_size
            )
        )
        for _ in range(1, n_blocks):
            layers.append(
                _BasicBlock(
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
        # print(o)
    t1 = perf_counter()
    print(f"{t1 - t0:.2f} seconds")
