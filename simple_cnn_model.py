import torch.nn as nn
from models.base_model import FCNet

class SimpleCNNModel(nn.Module):
    def __init__(self, in_channels, out_channels, output_dimension, dropout=0.2):
        super(SimpleCNNModel, self).__init__()

        self.output_dimension = output_dimension

        self.conv_expand_1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.batch_norm1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.batch_norm1_2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc_dimension = 7 * 7 * 512
        self.inner_fc_dim = 2048

        self.fc1 = nn.Linear(self.fc_dimension, self.inner_fc_dim)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.fc2 = nn.Linear(self.inner_fc_dim, self.inner_fc_dim)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc3 = nn.Linear(self.inner_fc_dim, self.output_dimension)

        self.relu = nn.ReLU()


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

        x = x.view(-1, self.fc_dimension)

        x = self.fc1(x)
        x = self.relu(x)
        self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        self.dropout2(x)

        x = self.fc3(x)


        return x