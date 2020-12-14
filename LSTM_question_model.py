
from torch import Tensor, nn
import torch
from functools import partial
from abc import ABCMeta




class LSTM(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, emb_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()

        self.lstm_model = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               bidirectional=True,
                               batch_first=False)



    def forward(self, x: Tensor) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        return self.lstm_model(x)
