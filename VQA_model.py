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


class VQA_model(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 128,
                 num_classes: int = 3219,
                 nhead: int = 4,
                 dropout: float = 0.2,
                 mean_with_attention: bool = True):

        super(VQA_model, self).__init__()


        self.word_emb_dim = word_emb_dim

        self.mean_with_attention = mean_with_attention

        self.num_classes = num_classes

        self.image_model = VGG19_E(3, self.word_emb_dim)

        self.question_model = TransofrmerEncoder(word_vocab_size=word_vocab_size,
                                                 word_emb_dim=self.word_emb_dim,
                                                 nhead = nhead)

        self.relu = nn.ReLU()

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.soft_max = nn.Softmax(dim=1)

        self.inner_fc_dim = 4096

        self.fc_dimension = 7 * 7 * 512
        self.inner_fc_dim = 4096

        layers_classifier = [
            weight_norm(nn.Linear(2*self.word_emb_dim, self.inner_fc_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(self.inner_fc_dim, self.num_classes), dim=None)
        ]
        self.classifier = nn.Sequential(*layers_classifier)



    def forward(self, input: (Tensor, Tensor, Tensor)) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        image, question, pad_mask = input
        batch_size = question.shape[0]
        seq_length = question.shape[1]

        question_outputs = self.question_model((question, pad_mask))

        image = image.squeeze(0)
        cnn_output = self.image_model(image)                                # [batch, output_dim_nets]

        if self.mean_with_attention:
            question_outputs = question_outputs.view(batch_size, seq_length, -1)
            # print('question_outputs', question_outputs.size())

            cnn_output = cnn_output.unsqueeze(-1)                                   # [batch, output_dim_nets, 1]
            # print('cnn_output', cnn_output.size())

            attention = torch.matmul(question_outputs, cnn_output).squeeze(-1)       # [batch, seq_length]
            # print('attention', attention.size())

            scalars = self.soft_max(attention)                                          # [batch, seq_length]
            # print('scalars', scalars.size())

            question_outputs = torch.matmul(scalars.unsqueeze(1), question_outputs).squeeze(1)  # [batch, output_dim_nets]
            # print('question_outputs', question_outputs.size())

            cnn_output = cnn_output.squeeze(-1)

        else:
            question_outputs = torch.mean(question_outputs, dim=0)


        mutual = torch.cat((question_outputs, cnn_output), dim=1)        # [batch, output_dim_nets]

        output = self.classifier(mutual)

        output = self.log_softmax(output)

        return output

