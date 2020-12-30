from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
from simple_cnn_model import SimpleCNNModel
from resnet import resnet18
from LSTM_question_model import LSTM
from VGG19_E import VGG19_E
import torch


class VQA_Attention(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, output_dim_nets: int = 1024,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 50,
                 num_classes: int = 3219,
                 LSTM_num_layers: int = 2,
                 dropout: float = 0.2):

        super(VQA_Attention, self).__init__()

        # self.image_model = resnet18(3, output_dim_nets)

        self.output_dim_nets = output_dim_nets
        self.num_classes = num_classes

        self.image_model = VGG19_E(3, self.output_dim_nets)

        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)

        self.LSTM_num_layers = LSTM_num_layers

        self.question_model = nn.LSTM(input_size=word_emb_dim, hidden_size=self.output_dim_nets//2, num_layers=LSTM_num_layers,
                               bidirectional=True,
                               batch_first=True)


        self.relu = nn.ReLU()

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.soft_max = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.inner_fc_dim = 4096

        self.fc1 = nn.Linear(self.output_dim_nets, self.inner_fc_dim)
        self.fc2 = nn.Linear(self.inner_fc_dim, self.inner_fc_dim)
        self.fc3 = nn.Linear(self.inner_fc_dim, self.num_classes)


    def forward(self, input: (Tensor, Tensor)) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        image, question = input
        batch_size = question.shape[0]
        seq_length = question.shape[1]

        # Pass word_idx and pos_idx through their embedding layers
        # print('question', question.size())
        word_vec = self.word_embedding(question)        # [batch, seq, emb_dim]
        # print('word vec', word_vec.size())

        output_lstm, (_, _) = self.question_model(word_vec)                # [batch_size, seq_len, output_dim_nets]
        # print('output_lstm', output_lstm.size())
        # print(output_lstm.shape)

        image = image.squeeze(0)
        # print('image', image.size(0))
        # print(image_path.shape)

        cnn_output = self.image_model(image)                                # [batch, output_dim_nets]
        # print('cnn_output', cnn_output.size())
        # print(cnn_output.shape)

        cnn_output = cnn_output.unsqueeze(-1)                               # [batch, output_dim_nets, 1]
        # print(cnn_output.shape)

        attention = torch.matmul(output_lstm, cnn_output).squeeze(-1)       # [batch, seq_length]
        # print('attention', attention.size())
        # print(attention.shape)

        scalars = self.soft_max(attention)                                  # [batch, seq_length]
        # print('scalars', scalars.size())

        # print(scalars.shape)

        lstm_to_multiplication = torch.matmul(scalars.unsqueeze(1), output_lstm).squeeze(1)  # [batch, output_dim_nets]
        # print('lstm_to_multiplication', lstm_to_multiplication.size())

        # print(lstm_to_multiplication.shape)

        mutual = lstm_to_multiplication * cnn_output.squeeze(-1)            # [batch, output_dim_nets]
        # print('mutual', mutual.size())

        # print(mutual.shape)

        # print(fc_output.shape)
        x = self.fc1(mutual)
        # print('fc1', x.size())

        x = self.relu(x)
        self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        self.dropout(x)

        x = self.fc3(x)
        # print('fc3_x', x.size())

        return self.log_softmax(x)

