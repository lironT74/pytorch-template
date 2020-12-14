import re
import unicodedata
import string
import torch

SOS_token = 0
EOS_token = 1
UKN_token = 2

def normalize_string(s):
    return s.lower().strip().translate(str.maketrans('', '', string.punctuation))


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<UKN>"}
        self.n_words = 3  # Count SOS and EOS and UNK

    def add_sentence(self, sentence):
        sentence = normalize_string(sentence)
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_embedding(self, sentence):
        sentence = normalize_string(sentence).split(' ')
        sentence_matrix = torch.zeros(len(sentence)+2, self.n_words)

        sentence_matrix[0][SOS_token] = 1
        sentence_matrix[len(sentence) + 1] = 1
        for i, word in enumerate(sentence):
            if word not in self.word2index:
                sentence_matrix[i+1][UKN_token] = 1
            else:
                sentence_matrix[i+1][self.word2index[word]] = 1

        # TODO: to add unsqueeze(0) for batch representation
        return sentence_matrix

