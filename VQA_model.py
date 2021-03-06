from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
from simple_cnn_model import SimpleCNNModel
from resnet import resnet18
from LSTM_question_model import LSTM
from VGG19_E import VGG19_E
from VGG19_A import VGG19_mini_A
import torch.nn.functional as f
import torch
from torch.nn.utils.weight_norm import weight_norm
from transormer_endocer import TransofrmerEncoder


class VQA_model(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 50,
                 num_classes: int = 3219,
                 nhead: int = 5,
                 dropout: float = 0.4,
                 weighted_matmul_mean: bool = True,
                 output_dim_nets: int = 1000,
                 LSTM_num_layers: int = 2):

        super(VQA_model, self).__init__()


        self.word_emb_dim = word_emb_dim

        self.mean_with_attention = weighted_matmul_mean

        self.num_classes = num_classes

        self.output_dim_nets = output_dim_nets


        self.question_model = LSTM(word_vocab_size=word_vocab_size,
                                   word_emb_dim=self.word_emb_dim,
                                   output_dim_nets=output_dim_nets,
                                   num_classes=num_classes,
                                   dropout=dropout,
                                   LSTM_num_layers=LSTM_num_layers)

        # self.question_model.load_state_dict(torch.load(f'/home/student/HW2/logs/only_LSTM_1_3_10_46_7/model.pth')['model_state'])

        # self.question_model = TransofrmerEncoder(word_vocab_size=word_vocab_size,
        #                                          word_emb_dim = self.word_emb_dim,
        #                                          nhead = nhead,
        #                                          output_dim_nets = output_dim_nets)

        #LOAD LSTM HERE

        self.image_model = VGG19_mini_A(3, output_dim_nets, dropout)

        self.relu = nn.ReLU()

        self.log_softmax = nn.LogSoftmax(dim=1)

        # self.soft_max = nn.Softmax(dim=1)

        self.inner_fc_dim = 4096


        # layers_classifier = [
        #     weight_norm(nn.Linear(output_dim_nets, self.inner_fc_dim), dim=None),
        #     nn.ReLU(),
        #     nn.Dropout(dropout, inplace=True),
        #     weight_norm(nn.Linear(self.inner_fc_dim, self.inner_fc_dim), dim=None),
        #     nn.ReLU(),
        #     nn.Dropout(dropout, inplace=True),
        #     weight_norm(nn.Linear(self.inner_fc_dim, self.num_classes), dim=None)
        # ]

        layers_classifier = [
            nn.Linear(output_dim_nets, self.num_classes),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(self.num_classes, self.num_classes),
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
            question_outputs = question_outputs.view(batch_size, -1, self.output_dim_nets)    # [batch, seq_len, output_dim_nets]
            # print('question_outputs', question_outputs.size())

            cnn_output = cnn_output.unsqueeze(-1)                                   # [batch, output_dim_nets, 1]
            # print('cnn_output', cnn_output.size())
            # print('question', question_outputs.size())
            question_image_matmul = torch.matmul(question_outputs, cnn_output).squeeze(-1)       # [batch, seq_length]

            #ignore <"PAD"> embeddings
            question_image_matmul[pad_mask == 1] = 0
            scalars = f.normalize(question_image_matmul, p=1, dim=1)

            question_outputs = torch.matmul(scalars.unsqueeze(1), question_outputs).squeeze(1)  # [batch, output_dim_nets]
            # print('question_outputs', question_outputs.size())

            cnn_output = cnn_output.squeeze(-1)

        else:

            #TODO: how to ignore <"PAD"> in this case?
            question_outputs = torch.mean(question_outputs, dim=0)


        # both = torch.cat((question_outputs, cnn_output), dim=1)        # [batch, output_dim_nets]

        both = question_outputs * cnn_output     # [batch, output_dim_nets]

        output = self.classifier(both)

        output = self.log_softmax(output)

        return output

