"""
    Example for a simple model
"""

from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
import torch


class MyModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, input_dim: int = 50, num_hid: int = 256, output_dim: int = 2, dropout: float = 0.2):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        return self.classifier(x)
