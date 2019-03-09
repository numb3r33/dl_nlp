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
