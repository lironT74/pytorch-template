import torch.nn as nn
import torch
from models.base_model import FCNet

class VGG19_A(nn.Module):
    def __init__(self, in_channels, output_dimension, return_before_fc=False):
        super(VGG19_A, self).__init__()

        self.output_dimension = output_dimension

        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc_dimension = 7 * 7 * 512
        self.inner_fc_dim = 4096

        self.fc1 = nn.Linear(self.fc_dimension, self.inner_fc_dim)
        self.fc2 = nn.Linear(self.inner_fc_dim, self.inner_fc_dim)
        self.fc3 = nn.Linear(self.inner_fc_dim, self.output_dimension)

        self.relu = nn.ReLU()
        self.return_before_fc = return_before_fc

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.pool(x)

        x_before_fc = x

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        if self.return_before_fc:
            return x, x_before_fc
        else:
            return x