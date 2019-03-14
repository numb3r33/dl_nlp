import config
import numpy as np
import torch

from utils import *

def get_neg_word_indices(word2index, batch_len, p_w):
    neg_words_indices = []

    for i in range(batch_len):
        neg_words = np.random.choice([k for k in word2index.keys()], size=config.NUM_NEG_WORDS, replace=False, p=[v for v in p_w.values()])
        neg_words_indices.append(np.array([word2index[w] for w in neg_words]))

    return neg_words_indices

def vanilla_neg_sampling_loss(logits, labels, word2index, p_w):
    neg_words_indices = get_neg_word_indices(word2index, len(labels), p_w)
    neg_words_indices = torch.cuda.LongTensor(neg_words_indices)

    neg_words = torch.gather(logits, 1, neg_words_indices)
    neg_sim   = torch.log(1 + torch.exp(neg_words))
    neg_sim   = torch.sum(neg_sim, dim=1)

    pos_sim   = logits[torch.cuda.LongTensor(np.arange(len(labels))), labels]
    pos_sim   = torch.log(1 + torch.exp(-pos_sim))

    loss      = torch.sum(pos_sim + neg_sim)

    return loss

def batch_transpose_trick_loss(center_word_repr, labels):
    x = 0.0

    for batch_index in range(len(center_word_repr)):
        w = center_word_repr[batch_index, :, :]
        x = x + torch.sum(torch.mm(w, torch.transpose(w, 1, 0)) - 1)

    y = 0.0

    for window_index in range(center_word_repr.shape[1]):
        b = center_word_repr[:, window_index, :]
        y = y + torch.sum(torch.mm(b, torch.transpose(b, 1, 0)))

    loss = -x + y

    return loss

def fasttext_loss_fn(logits, labels, batch_len, context_window, word2index, p_w):
    pos_word_indices = get_pos_word_indices(batch_len, context_window, labels)
    pos_word_indices = torch.cuda.LongTensor(pos_word_indices)

    pos_sim = torch.gather(logits, 2, pos_word_indices)
    pos_sim = torch.log(1 + torch.exp(-pos_sim))
    pos_sim = torch.sum(torch.sum(pos_sim, dim=2), dim=1)

    neg_word_indices = get_neg_word_indices_ft(batch_len, context_window, word2index, p_w)
    neg_word_indices = torch.cuda.LongTensor(neg_word_indices)

    neg_sim = torch.gather(logits, 2, neg_word_indices)
    neg_sim = torch.log(1 + torch.exp(neg_sim))
    neg_sim = torch.sum(torch.sum(neg_sim, dim=2), dim=1)

    loss    = torch.sum(pos_sim + neg_sim)

    return loss
