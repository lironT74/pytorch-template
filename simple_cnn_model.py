import torch.nn as nn
from models.base_model import FCNet

class SimpleCNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, output_dimension):
        super(SimpleCNNModel, self).__init__()

        self.output_dimension = output_dimension

        self.conv1 = nn.Conv2d(in_channels, 6, 3, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, out_channels, 3, 1, 2)
        self.pool_avg_pool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc_dimension = out_channels * 5 * 5

        self.fc1 = nn.Linear(self.fc_dimension, self.output_dimension)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.pool_avg_pool(self.relu(self.conv2(x)))

        x = x.view(-1, self.fc_dimension)

        x = self.relu(self.fc1(x))

        return x