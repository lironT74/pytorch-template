import torch.nn as nn
import torch
from models.base_model import FCNet

class mini_VGG19(nn.Module):
    def __init__(self, in_channels, out_channels, output_dimension):
        super(mini_VGG19, self).__init__()

        self.output_dimension = output_dimension

        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.fc_dimension = 7 * 7 * 512

        self.fc1 = nn.Linear(self.fc_dimension, self.output_dimension)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        fc_output = self.fc1(x)

        return self.relu(fc_output)