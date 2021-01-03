
from torch import Tensor, nn
import torch
from functools import partial
from abc import ABCMeta

from torch.nn.utils.weight_norm import weight_norm


class LSTM(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, output_dim_nets,
                 word_vocab_size,
                 word_emb_dim,
                 dropout,
                 LSTM_num_layers,
                 num_classes
                 ):

        super(LSTM, self).__init__()


        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size,
                                           embedding_dim=word_emb_dim)


        self.lstm_model = nn.LSTM(input_size=word_emb_dim,
                                  hidden_size=output_dim_nets//2,
                                  num_layers=LSTM_num_layers,
                                  bidirectional=True,
                                  batch_first=True)

        self.LSTM_num_layers = LSTM_num_layers

        self.num_classes = num_classes
        self.inner_fc_dim = 4096

        layers_classifier = [
            weight_norm(nn.Linear(output_dim_nets, self.inner_fc_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(self.inner_fc_dim, self.inner_fc_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(self.inner_fc_dim, self.num_classes), dim=None)
        ]

        self.classifier = nn.Sequential(*layers_classifier)

        self.log_softmax = nn.LogSoftmax(dim=1)



    def forward(self, input_question) -> Tensor:

        question, _ = input_question
        batch_size = question.shape[0]

        word_vec = self.word_embedding(question)

        output, (h_n, _) = self.lstm_model(word_vec)

        return output

        # h_n = h_n.view(self.LSTM_num_layers, batch_size, -1)[-1]  # [batch, output_dim_nets]
        #
        # output = self.classifier(h_n)
        #
        # output = self.log_softmax(output)
        #
        # return output


