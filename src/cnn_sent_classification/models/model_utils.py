import torch

from .cnn_model import CNNModel
from .simple_cnn import SimpleCNN
from .rcnn import RCNN

__all__ = ['CNNModel', 'SimpleCNN', 'RCNN']


def make_model(config, emb_matrix):
    vocab_size = config['vocab_size']
    embed_size = config['embed_size']
    pad_idx    = config['pad_idx']

    return globals()[config['model_name']](torch.FloatTensor(emb_matrix), vocab_size, embed_size, config, pad_idx)
