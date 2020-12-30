from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
from simple_cnn_model import SimpleCNNModel
from resnet import resnet18
from LSTM_question_model import LSTM
from VGG19_E import VGG19_E
import torch
from torch.nn.utils.weight_norm import weight_norm
from transormer_endocer import TransofrmerEncoder


class VQA_Attention(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 512,
                 num_classes: int = 3219,
                 nhead: int = 4,
                 dropout: float = 0.2):

        super(VQA_Attention, self).__init__()

        # self.image_model = resnet18(3, output_dim_nets)

        self.word_emb_dim = word_emb_dim
        self.num_classes = num_classes

        self.image_model = VGG19_E(3, self.word_emb_dim)


        # self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)
        # self.LSTM_num_layers = LSTM_num_layers
        # self.question_model = nn.LSTM(input_size=word_emb_dim, hidden_size=self.output_dim_nets//2, num_layers=LSTM_num_layers,
        #                        bidirectional=True,
        #                        batch_first=True)

        self.question_model = TransofrmerEncoder(word_vocab_size=word_vocab_size, word_emb_dim=word_emb_dim, nhead = nhead)

        self.relu = nn.ReLU()

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.soft_max = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.inner_fc_dim = 2048

        self.fc1 = weight_norm(nn.Linear(2*self.word_emb_dim, self.inner_fc_dim), dim=None)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.fc2 = weight_norm(nn.Linear(self.inner_fc_dim, self.inner_fc_dim), dim=None)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc3 = weight_norm(nn.Linear(self.inner_fc_dim, self.num_classes), dim=None)


    def forward(self, input: (Tensor, Tensor, Tensor)) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        image, question, pad_mask = input
        batch_size = question.shape[0]
        seq_length = question.shape[1]

        # Pass word_idx and pos_idx through their embedding layers

        # output_lstm, (_, _) = self.question_model(word_vec)                # [batch_size, seq_len, output_dim_nets]
        # # print('output_lstm', output_lstm.size())
        # # print(output_lstm.shape)

        question_outputs = self.question_model((question, pad_mask))
        question_outputs = question_outputs.view(batch_size, seq_length, -1)

        image = image.squeeze(0)
        # print('image', image.size(0))
        # print(image_path.shape)

        cnn_output = self.image_model(image)                                # [batch, output_dim_nets]
        # print('cnn_output', cnn_output.size())
        # print(cnn_output.shape)

        cnn_output = cnn_output.unsqueeze(-1)                               # [batch, output_dim_nets, 1]
        # print(cnn_output.shape)

        attention = torch.matmul(question_outputs, cnn_output).squeeze(-1)       # [batch, seq_length]
        # print('attention', attention.size())
        # print(attention.shape)

        scalars = self.soft_max(attention)                                  # [batch, seq_length]
        # print('scalars', scalars.size())

        # print(scalars.shape)

        question_to_multiplication = torch.matmul(scalars.unsqueeze(1), question_outputs).squeeze(1)  # [batch, output_dim_nets]
        # print('lstm_to_multiplication', lstm_to_multiplication.size())
        # print(lstm_to_multiplication.shape)



        cnn_output = cnn_output.squeeze(-1)

        mutual = torch.cat((question_to_multiplication, cnn_output), dim=1)        # [batch, output_dim_nets]


        x = self.fc1(mutual)
        self.dropout1(x)
        x = self.relu(x)


        x = self.fc2(x)
        self.dropout2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # print('fc3_x', x.size())

        return self.log_softmax(x)

