import torch.nn as nn
import torch
from models.base_model import FCNet

class VGG19_E(nn.Module):
    def __init__(self, in_channels, output_dimension, dropout=0.2, return_before_fc=False):
        super(VGG19_E, self).__init__()

        self.output_dimension = output_dimension

        self.conv1_1 = nn.Conv2d(in_channels, 64, 9, 1, 4)
        self.conv1_2 = nn.Conv2d(64, 64, 9, 1, 4)

        self.conv2_1 = nn.Conv2d(64, 128, 7, 1, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 7, 1, 3)

        self.conv3_1 = nn.Conv2d(128, 256, 5, 1, 2)
        self.conv3_2 = nn.Conv2d(256, 256, 5, 1, 2)
        self.conv3_3 = nn.Conv2d(256, 256, 5, 1, 2)
        self.conv3_4 = nn.Conv2d(256, 256, 5, 1, 2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc_dimension = 7 * 7 * 512
        self.inner_fc_dim = 4096

        self.fc1 = nn.Linear(self.fc_dimension, self.inner_fc_dim)
        self.fc2 = nn.Linear(self.inner_fc_dim, self.inner_fc_dim)
        self.fc3 = nn.Linear(self.inner_fc_dim, self.output_dimension)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.return_before_fc = return_before_fc

    def forward(self, x):

        saved_x = x
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = x + saved_x
        x = self.pool(x)

        saved_x = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = x + saved_x
        x = self.pool(x)

        saved_x = x
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = x + saved_x
        saved_x = x
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.relu(x)
        x = x + saved_x
        x = self.pool(x)

        saved_x = x
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = saved_x + x
        saved_x = x
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.conv4_4(x)
        x = self.relu(x)
        x = saved_x + x
        x = self.pool(x)

        saved_x = x
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = saved_x + x
        saved_x = x
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.conv5_4(x)
        x = self.relu(x)
        x = saved_x + x
        x = self.pool(x)

        x_before_fc = x

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        self.dropout(x)

        x = self.fc3(x)

        if self.return_before_fc:
            return x, x_before_fc
        else:
            return x