
from torch import Tensor, nn
import torch
from functools import partial
from abc import ABCMeta

from torch.nn.utils.weight_norm import weight_norm


class LSTM(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, output_dim_nets: int = 1024,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 128,
                 dropout: float = 0.2,
                 LSTM_num_layers: int = 3):

        super(LSTM, self).__init__()


        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size,
                                           embedding_dim=word_emb_dim)


        self.lstm_model = nn.LSTM(input_size=word_emb_dim,
                                  hidden_size=output_dim_nets//2,
                                  num_layers=LSTM_num_layers,
                                  bidirectional=True,
                                  batch_first=True)

        self.LSTM_num_layers = LSTM_num_layers


    def forward(self, input_question) -> Tensor:

        question, pad_mask = input_question

        word_vec = self.word_embedding(question)

        output, (_, _) = self.lstm_model(word_vec)


        return output
