
from torch import Tensor, nn
from abc import ABCMeta


class TransofrmerEncoder(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self,
                 word_vocab_size: int = 100000,
                 word_emb_dim: int = 128,
                 nhead: int= 4):

        super(TransofrmerEncoder, self).__init__()


        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)
        self.endocer_layer = nn.TransformerEncoderLayer(d_model=word_emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.endocer_layer, num_layers=4)


    def forward(self, input: (Tensor, Tensor)) -> Tensor:
        question, pad_mask = input
        batch_size = question.shape[0]
        seq_length = question.shape[1]

        word_vec = self.word_embedding(question)
        word_vec = word_vec.view(seq_length, batch_size, -1)

        output = self.transformer_encoder(word_vec, src_key_padding_mask = pad_mask)

        return output
