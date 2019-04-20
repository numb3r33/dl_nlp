import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import gc
import gensim.models.keyedvectors as word2vec
from exp_config import config

__all__ = ['check_labels', 'load_w2v_embedding', 'pad_collate']


def check_labels(y): return all(v is None for v in y)


def load_w2v_embedding(emb_matrix, vocab, embed_size):
    word2vec_dict   = word2vec.KeyedVectors.load_word2vec_format(config['w2v_path'], binary=True)
    embedding_index = dict()
    
    for word in word2vec_dict.wv.vocab:
        embedding_index[word] = word2vec_dict.word_vec(word)

    embed_cnt = 0

    for i, word in enumerate(vocab.itos):
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            emb_matrix[i] = embedding_vector
            embed_cnt = embed_cnt + 1

    del embedding_index
    gc.collect()

    # fill pad token with all zeros
    emb_matrix[vocab.stoi[config['PAD']]] = np.zeros(embed_size)
    print('total embedded {} common words'.format(embed_cnt))
    
    return emb_matrix


def load_fasttext_embedding(emb_matrix, vocab, embed_size):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype=np.float32)
    
    EMB_FILE        = config['fasttext_path']
    embedding_index = dict()
    
    with open(EMB_FILE, 'r') as f:
        for o in f.readlines()[1:]:
            word, coefs = get_coefs(*o.rstrip().rsplit(' '))
            embedding_index[word] = coefs

    print('Loaded {} word vectors'.format(len(embedding_index)))

    embed_cnt = 0

    for i, word in enumerate(vocab.itos):
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            emb_matrix[i] = embedding_vector
            embed_cnt = embed_cnt + 1

    del embedding_index
    gc.collect()

    # fill pad token with all zeros
    emb_matrix[vocab.stoi[config['PAD']]] = np.zeros(embed_size)
    print('total embedded {} common words'.format(embed_cnt))
    
    return emb_matrix


def pad_collate(data, pad_idx, sent_len):
    if len(data) == 1:
        sequences, labels = data[0]
        sequences = sequences.view(1, -1)
        if labels is not None: labels = labels.view(1, -1)
    else:
        sequences, labels = zip(*data)
        if not check_labels(labels): labels = torch.cat([l.view(-1, 1) for l in labels], dim=1).t()
        sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    
    sent_len  = min(sequences.size(1), sent_len)
    sequences = sequences[:, :sent_len]
    
    return sequences, labels


def read_csv(fn): return pd.read_csv(fn)
