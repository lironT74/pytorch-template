from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor

from resnet import resnet18
from LSTM_question_model import LSTM




class VQA(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, output_dim_nets: int = 1024,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 100,
                 num_classes: int = 3219,
                 LSTM_num_layers: int = 4):


        super(VQA, self).__init__()


        self.image_model = resnet18(3, output_dim_nets)


        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)

        self.question_model = nn.LSTM(input_size=word_emb_dim, hidden_size=output_dim_nets//2, num_layers=LSTM_num_layers,
                               bidirectional=True,
                               batch_first=True)

        self.fc = nn.Linear(output_dim_nets, num_classes)

        self.log_softmax = nn.LogSoftmax()



    def forward(self, input: (Tensor, Tensor)) -> Tensor:
        print('fuck my dick')
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        image, question = input
        batch_size = question.shape[0]
        seq_length = question.shape[1]

        # Pass word_idx and pos_idx through their embedding layers
        word_vec = self.word_embedding(question)        # [batch, seq, emb_dim]

        print('word embd size:', word_vec.size())
        print(self.question_model(word_vec).size())
        lstm_output = self.question_model(word_vec)[range(batch_size), [-1]*seq_length]              # [batch, output_dim_nets]

        resnet_output = self.image_model(image)                         # [batch, output_dim_nets]

        mutual = lstm_output * resnet_output                            # [batch, output_dim_nets]

        fc_output = self.fc(mutual)                                     # [batch, num_classes]

        return self.log_softmax(fc_output)



if __name__ == '__main__':
    VQA()