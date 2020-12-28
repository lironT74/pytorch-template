
from torch import Tensor, nn
import torch
from functools import partial
from abc import ABCMeta




class LSTM(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, output_dim_nets: int = 1024,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 50,
                 num_classes: int = 3219,
                 LSTM_num_layers: int = 2):
        super(LSTM, self).__init__()

        self.lstm_model = nn.LSTM(input_size=word_emb_dim, hidden_size=output_dim_nets//2, num_layers=LSTM_num_layers,
                               bidirectional=True,
                               batch_first=True)
        self.LSTM_num_layers = LSTM_num_layers

        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)



        self.fc_dimension = output_dim_nets
        # self.inner_fc_dim = 1000
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.fc_dimension, self.fc_dimension)
        self.fc2 = nn.Linear(self.fc_dimension, self.fc_dimension)
        self.fc3 = nn.Linear(self.fc_dimension, self.num_classes)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, question) -> Tensor:
        word_vec = self.word_embedding(question)
        batch_size = word_vec.shape[0]
        _, (h_n_lstm, _) = self.lstm_model(word_vec)


        h_n_lstm = h_n_lstm.view(self.LSTM_num_layers, batch_size, -1)[-1]  # [batch, output_dim_nets]

        x = self.fc1(h_n_lstm)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return self.log_softmax(x)
