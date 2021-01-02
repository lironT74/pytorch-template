import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm

class VGG19_mini_A(nn.Module):
    def __init__(self, in_channels, output_dimension, dropout=0.2, return_before_fc=False):
        super(VGG19_mini_A, self).__init__()

        self.output_dimension = output_dimension

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        # self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        # self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc_dimension = 7 * 7 * 512
        self.inner_fc_dim = 2048

        layers_classifier = [
            weight_norm(nn.Linear(self.fc_dimension, self.inner_fc_dim), dim=None),
            nn.ReLU(),

            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(self.inner_fc_dim, self.output_dimension), dim=None)
        ]
        self.classifier = nn.Sequential(*layers_classifier)


        self.return_before_fc = return_before_fc
        self.relu = nn.ReLU()


    def forward(self, x):


        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.pool(x)


        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.pool(x)


        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.pool(x)


        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.pool(x)


        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.pool(x)


        x_before_fc = x
        x = x.view(-1, self.fc_dimension)
        x = self.classifier(x)


        if self.return_before_fc:
            return x, x_before_fc
        else:
            return x