from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
from simple_cnn_model import SimpleCNNModel
from resnet import resnet18
from LSTM_question_model import LSTM
from VGG19_A import VGG19_A
import torch


class VQA_from_lecture(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, output_dim_nets: int = 1024,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 50,
                 num_classes: int = 3219,
                 LSTM_num_layers: int = 2,
                 max_sentence_length: int = 19,
                 d_for_interaction: int = 100):

        super(VQA_from_lecture, self).__init__()

        # self.image_model = resnet18(3, output_dim_nets)

        self.image_model = VGG19_A(3, output_dim_nets, return_before_fc=True)
        self.psi_i_V_i = nn.Parameter(torch.randn(1, 49, 512, 512))
        self.psi_i_v_i = nn.Parameter(torch.randn(1, 49, 512, 1))


        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)

        self.LSTM_num_layers = LSTM_num_layers

        self.question_model = nn.LSTM(input_size=word_emb_dim, hidden_size=output_dim_nets//2, num_layers=LSTM_num_layers,
                               bidirectional=True,
                               batch_first=True)
        self.max_sentence_length = max_sentence_length
        # self.psi_i_Q_i = nn.ParameterList([nn.Parameter(torch.randn(output_dim_nets, output_dim_nets)) for _ in range(max_sentence_length)])
        # self.psi_i_q_i = nn.ParameterList([nn.Parameter(torch.randn(output_dim_nets, 1)) for _ in range(max_sentence_length)])
        self.psi_i_Q_i = nn.Parameter(torch.randn(1, max_sentence_length, output_dim_nets, output_dim_nets))
        self.psi_i_q_i = nn.Parameter(torch.randn(1, max_sentence_length, output_dim_nets, 1))

        self.w_i = nn.Parameter(torch.randn(1))
        self.w_i_i = nn.Parameter(torch.randn(1))
        self.w_i_q = nn.Parameter(torch.randn(1))
        self.w_q = nn.Parameter(torch.randn(1))
        self.w_q_q = nn.Parameter(torch.randn(1))
        self.w_q_i = nn.Parameter(torch.randn(1))

        self.weights_mu_image_question = nn.Linear(max_sentence_length, 1)
        self.weights_mu_question_image = nn.Linear(49, 1)

        self.weights_mu_image_image = nn.Linear(49, 1)
        self.weights_mu_question_question = nn.Linear(max_sentence_length, 1)


        self.d_for_interaction = d_for_interaction

        # Previous Ri_q, Lq_i: for every region and word exists unique Ri_q and Lq_i
        # self.Ri_q = nn.Parameter(torch.randn(1, 49, 512, d_for_interaction))
        # self.Lq_i = nn.Parameter(torch.randn(1, max_sentence_length, output_dim_nets, d_for_interaction))
        self.Ri_q = nn.Linear(512, d_for_interaction, bias=False)
        self.Lq_i = nn.Linear(output_dim_nets, d_for_interaction, bias=False)

        # new Ri_q, Lq_i

        self.relu = nn.ReLU()
        self.soft_max = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc_dimension = output_dim_nets+512
        self.inner_fc_dim = 4096

        self.fc1 = nn.Linear(self.fc_dimension, self.inner_fc_dim)
        self.fc2 = nn.Linear(self.inner_fc_dim, self.inner_fc_dim)
        self.fc3 = nn.Linear(self.inner_fc_dim, self.output_dimension)



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
        word_vec = self.word_embedding(question)        # [batch, seq, emb_dim]
        # print(word_vec.size())
        # added_tensors = torch.zeros(word_vec.shape[0], self.max_sentence_length - seq_length,
        #                             word_vec.shape[-1])  # [batch, seq_len, 1, emb_dim]
        # # print(added_tensors.size())
        # if torch.cuda.is_available():
        #     added_tensors = added_tensors.cuda()
        #
        # word_vec = torch.cat((word_vec, added_tensors), 1)  # [batch, max_len, 1, output_dim_nets]

        lstm_outputs, (_, _) = self.question_model(word_vec)                # outpup_lstm = [batch_size, seq_len, output_dim_nets]
        seq_len = lstm_outputs.shape[1]
        lstm_outputs = lstm_outputs.view(lstm_outputs.shape[0], seq_len, 1, -1)                         # [batch, seq_len, 1, output_dim_nets]
        # print(output_lstm.shape)

        image = image.squeeze(0)
        # print(image_path.shape)

        _, image_regions = self.image_model(image)                                # [batch, output_dim_nets], [batch, region_dim=512 ,7, 7]

        # print('from cnn' ,image_regions)
        image_regions = image_regions.view(image_regions.shape[0],
                                           image_regions.shape[2]*image_regions.shape[3], 1, -1)  # [batch, 49, 1, region_dim=512]
        # print('after review', image_regions)


        psi_i = self.get_psi_i(image_regions)                                     # [batch, num_of_regions=49]
        psi_q = self.get_psi_q(lstm_outputs)                                 # [batch, max_len]
        # print('line 94', psi_i)

        mu_image_question, mu_question_image = self.get_mu_q_i(lstm_outputs, image_regions, seq_len)    # [batch, num_of_rehions=49], [batch, max_len]
        mu_i_i, mu_q_q = self.get_mu_i_mu_q(image_regions, lstm_outputs)




        b_i = self.w_i*psi_i + self.w_i_i*mu_i_i + self.w_i_q*mu_image_question             # [batch, num_of_rehions=49]
        b_q = self.w_q*psi_q + self.w_q_q*mu_q_q + self.w_q_i*mu_question_image             # [batch, max_len]

        # b_i = psi_i + mu_image_question  # [batch, num_of_rehions=49]
        # b_q = psi_q + mu_question_image  # [batch, max_len]


        softmaxed_image = self.soft_max(b_i).unsqueeze(1)     # [batch, 1, max_len=49]
        softmaxed_question = self.soft_max(b_q).unsqueeze(1)     # [batch, 1, num_of_rehions]
        # print('line 101', softmaxed_image)


        a_i = torch.matmul(softmaxed_image, image_regions.squeeze(2)).squeeze(1)       # [batch, region_dim=512]
        a_q = torch.matmul(softmaxed_question, lstm_outputs.squeeze(2)).squeeze(1)     # [batch, output_dim_nets]


        final_vec = torch.cat((a_i, a_q), dim=1)  # [batch, region_dim=512 + output_dim_nets]

        x = self.fc1(final_vec)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return self.log_softmax(x)

    def get_psi_i(self, image_regions):
        output = torch.matmul(image_regions, self.psi_i_V_i)           # [batch, num_of_rehions=49, 1, region_dim=512]
        output = self.relu(output)
        output = torch.matmul(output, self.psi_i_v_i)                  # [batch, num_of_rehions=49, 1, 1]
        output = output.squeeze(-1).squeeze(-1)                        # [batch, num_of_regions=49]
        return output

    def get_psi_q(self, lstm_outputs):
        output = torch.matmul(lstm_outputs, self.psi_i_Q_i)  # [batch, max_len, 1, output_dim_nets]
        output = self.relu(output)
        output = torch.matmul(output, self.psi_i_q_i)  # [batch, max_len, 1, 1]
        output = output.squeeze(-1).squeeze(-1)  # [batch, max_len]
        return output

    def get_mu_q_i(self, lstm_outputs, image_regions, seq_len):

        # image_non_normalized = torch.matmul(image_regions, self.Ri_q).squeeze(2)
        image_non_normalized = self.Ri_q(image_regions)  # [batch, num_of_rehions=49, d_for_interaction]

        # q_vecs_non_normalized = torch.matmul(lstm_outputs, self.Lq_i).squeeze(2)
        q_vecs_non_normalized = self.Lq_i(lstm_outputs)  # [batch, max_len, d_for_interaction]

        image_norms = torch.norm(image_non_normalized, dim=-1).unsqueeze(-1)
        image_norms = image_norms.expand_as(image_non_normalized) # [batch, num_of_rehions=49, d_for_interaction]

        q_vecs_norms = torch.norm(q_vecs_non_normalized, dim=-1).unsqueeze(-1)
        q_vecs_norms = q_vecs_norms.expand_as(q_vecs_non_normalized) # [batch, max_len, d_for_interaction]

        image_normalized = image_non_normalized / image_norms                       # [batch, num_of_rehions=49, d_for_interaction]

        q_vecs_normalized = q_vecs_non_normalized / q_vecs_norms

        psi_q_i = torch.matmul(q_vecs_normalized, torch.transpose(image_normalized, -2, -1)) # [batch, max_len, num_of_rehions=49]

        # mu_image_question = torch.sum(psi_q_i, 1)                                   # [batch, num_of_rehions=49]
        # mu_question_image = torch.sum(psi_q_i, 2)                                   # [batch, max_len]

        mu_image_question = self.weights_mu_image_question(torch.transpose(psi_q_i, -2, -1)).squeeze(-1) # [batch, num_of_rehions=49]
        mu_question_image = self.weights_mu_question_image(psi_q_i).squeeze(-1)  # [batch, num_of_rehions=49]


        return mu_image_question, mu_question_image

    def get_mu_i_mu_q(self, image_regions, q_vecs):

        image_norms = torch.norm(image_regions, dim=-1).unsqueeze(-1)
        image_norms = image_norms.expand_as(image_norms)  # [batch, num_of_rehions=49, 512]

        q_vecs_norms = torch.norm(q_vecs, dim=-1).unsqueeze(-1)
        q_vecs_norms = q_vecs_norms.expand_as(q_vecs)  # [batch, max_len, d_for_interaction]

        image_normalized = image_regions / image_norms  # [batch, num_of_rehions=49, 512]

        q_vecs_normalized = q_vecs / q_vecs_norms

        psi_i_i = torch.matmul(image_normalized, torch.transpose(image_normalized, -2, -1)) # [batch, num_of_rehions, num_of_rehions=49]

        psi_q_q = torch.matmul(q_vecs_norms, torch.transpose(q_vecs_norms, -2, -1)) # [batch, max_len, max_len=49]

        # mu_i_i = torch.sum(psi_i_i, 1)  # [batch, num_of_rehions=49]
        # mu_q_q = torch.sum(psi_q_q, 1)  # [batch, max_len]

        mu_i_i = self.weights_mu_image_image(psi_i_i).squeeze(-1)
        mu_q_q = self.weights_mu_question_question(psi_q_q).squeeze(-1)

        return mu_i_i, mu_q_q













