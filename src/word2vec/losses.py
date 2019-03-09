import config
import numpy as np
import torch

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
