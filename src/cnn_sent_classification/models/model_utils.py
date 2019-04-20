import torch

from .cnn_model import CNNModel

__all__ = ['CNNModel']


def make_model(config):
    emb_matrix = config['emb_matrix']
    vocab_size = config['vocab_size']
    embed_size = config['embed_size']

    return globals()[config['model_name']](torch.FloatTensor(emb_matrix), vocab_size, embed_size, config)
