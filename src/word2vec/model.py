import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config


def create_model(V):
    model = nn.Sequential(nn.Embedding(V, config.EMBEDDING_SIZE),
                          nn.Linear(config.EMBEDDING_SIZE, V)
            ).cuda()

    return model

def create_model_v2(V):
    model = nn.Sequential(nn.Embedding(V, config.EMBEDDING_SIZE)).cuda()

    return model

class SISG(nn.Module):
    def __init__(self, vocab_len, ngram_hash):
        super(SISG, self).__init__()

        self.embedding = nn.Embedding(len(ngram_hash) + 1, 32, padding_idx=0)
        self.linear    = nn.Linear(32, vocab_len)

    def forward(self, x):
        char_embeddings = self.embedding(x)
        word_embeddings = torch.sum(char_embeddings, dim=2)
        out             = self.linear(word_embeddings)

        return out
