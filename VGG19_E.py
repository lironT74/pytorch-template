import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm

from models.base_model import FCNet

class VGG19_E(nn.Module):
    def __init__(self, in_channels, output_dimension, dropout=0.2, return_before_fc=False):
        super(VGG19_E, self).__init__()

        self.output_dimension = output_dimension

        self.conv_expand_1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
        self.conv1_1 = nn.Conv2d(in_channels, 64, 9, 1, 4)
        self.batch_norm1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 9, 1, 4)
        self.batch_norm1_2 = nn.BatchNorm2d(64)


        self.conv_expand_2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.conv2_1 = nn.Conv2d(64, 128, 7, 1, 3)
        self.batch_norm2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 7, 1, 3)
        self.batch_norm2_2 = nn.BatchNorm2d(128)


        self.conv_expand_3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.conv3_1 = nn.Conv2d(128, 256, 5, 1, 2)
        self.batch_norm3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 5, 1, 2)
        self.batch_norm3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 5, 1, 2)
        self.batch_norm3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 5, 1, 2)
        self.batch_norm3_4 = nn.BatchNorm2d(256)


        self.conv_expand_4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.batch_norm4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm4_4 = nn.BatchNorm2d(512)


        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm5_4 = nn.BatchNorm2d(512)


        self.pool = nn.MaxPool2d(2, 2)


        self.fc_dimension = 7 * 7 * 512
        self.inner_fc_dim = 2048

        self.fc1 = weight_norm(nn.Linear(self.fc_dimension, self.inner_fc_dim), dim=None)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.fc2 = weight_norm(nn.Linear(self.inner_fc_dim, self.inner_fc_dim), dim=None)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc3 = weight_norm(nn.Linear(self.inner_fc_dim, self.output_dimension), dim=None)

        self.relu = nn.ReLU()
        self.return_before_fc = return_before_fc

    def forward(self, x):

        saved_x = self.conv_expand_1(x)
        x = self.conv1_1(x)
        x = self.batch_norm1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.batch_norm1_2(x)
        x = self.relu(x)
        x = x + saved_x
        x = self.pool(x)


        saved_x = self.conv_expand_2(x)
        x = self.conv2_1(x)
        x = self.batch_norm2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.batch_norm2_2(x)
        x = self.relu(x)
        x = x + saved_x
        x = self.pool(x)


        saved_x = self.conv_expand_3(x)
        x = self.conv3_1(x)
        x = self.batch_norm3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.batch_norm3_2(x)
        x = self.relu(x)
        x = x + saved_x
        saved_x = x
        x = self.conv3_3(x)
        x = self.batch_norm3_3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.batch_norm3_4(x)
        x = self.relu(x)
        x = x + saved_x
        x = self.pool(x)


        saved_x = self.conv_expand_4(x)
        x = self.conv4_1(x)
        x = self.batch_norm4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.batch_norm4_2(x)
        x = self.relu(x)
        x = saved_x + x
        saved_x = x
        x = self.conv4_3(x)
        x = self.batch_norm4_3(x)
        x = self.relu(x)
        x = self.conv4_4(x)
        x = self.batch_norm4_4(x)
        x = self.relu(x)
        x = saved_x + x
        x = self.pool(x)



        saved_x = x
        x = self.conv5_1(x)
        x = self.batch_norm5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.batch_norm5_2(x)
        x = self.relu(x)
        x = saved_x + x
        saved_x = x
        x = self.conv5_3(x)
        x = self.batch_norm5_3(x)
        x = self.relu(x)
        x = self.conv5_4(x)
        x = self.batch_norm5_4(x)
        x = self.relu(x)
        x = saved_x + x
        x = self.pool(x)


        x_before_fc = x


        x = torch.flatten(x, start_dim=1)


        x = self.fc1(x)
        x = self.relu(x)
        self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        self.dropout2(x)


        x = self.fc3(x)


        if self.return_before_fc:
            return x, x_before_fc
        else:
            return x