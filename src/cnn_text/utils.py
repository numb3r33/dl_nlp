import pandas as pd
import numpy as np
import os
from config import *

from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

tokenizer = WordPunctTokenizer()

def read_csv(fp):
    print('Reading file from: {}'.format(fp))
    return pd.read_csv(fp)

def get_target_cols():
    return  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'decent']

def create_new_target(df):
    print('Creating new target field')
    TARGET_COLS         =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] 
    df.loc[:, 'decent'] = 1 - df.loc[:, TARGET_COLS].max(axis=1)

    return df

def get_tokenized_comments(comment):
    print('Create tokenized comments')
    return list(map(tokenizer.tokenize, comment))

def get_tokenized_comments_string(comment):
    print('Create tokenized comments ( str )')
    return list(map(' '.join, map(tokenizer.tokenize, comment)))

def define_word2vec(tokenized_comments, exp_name, model_fp):
    print('Defining word2vec model')

    if os.path.exists(model_fp):
        model = joblib.load(model_fp)
    else:
        model = Word2Vec(tokenized_comments,
                         size=PARAMS[exp_name]['EMBEDDING_SIZE'],
                         min_count=PARAMS[exp_name]['MIN_FREQ'],
                         window=PARAMS[exp_name]['CONTEXT_WINDOW']
                        ).wv

        joblib.dump(model, model_fp)

    return model

def get_word2vec_vocab(model):
    print('Fetch Task vocabulary')
    return sorted(model.vocab.keys(), key=lambda word: model.vocab[word].count, reverse=True)

def get_word_vectors(model, words):
    print('Word vectors for all the tokens')
    return model.vectors[[model.vocab[word].index for word in words]]

def get_mean_std_embedding(word_vectors):
    print('Calculate mean and std of word embeddings')
    emb_mean, emb_std = word_vectors.mean(), word_vectors.std()
    return emb_mean, emb_std

def create_embedding(word_vectors, words, word2vec, emb_mean, emb_std, exp_name):
    print('Create word embeddings')

    UNK, PAD       = 'UNK', 'PAD'
    UNK_IX, PAD_IX = len(words), len(words) + 1
    nb_words       = len(words) + 2

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, PARAMS[exp_name]['EMBEDDING_SIZE']))

    for word in words + [UNK, PAD]:
        if word in word2vec.vocab:
            word_idx                   = word2vec.vocab[word].index
            embedding_vector           = word2vec.vectors[word2vec.vocab[word].index]
            embedding_matrix[word_idx] = embedding_vector

    return embedding_matrix, UNK_IX, PAD_IX

def get_token_to_id(words, word2vec, UNK_IX, PAD_IX):
    print('Prepare token to id mapping')

    token_to_id = {word: word2vec.vocab[word].index for word in words}
    token_to_id[UNK_IX] = UNK_IX
    token_to_id[PAD_IX] = PAD_IX

    return token_to_id

def as_matrix(sequences, token_to_id, UNK_IX, PAD_IX, max_len=None):
    """ Convert a list of tokens into a matrix with padding """

    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))

    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))

    for i,seq in enumerate(sequences):
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix


def get_train_test_splits(train, exp_name):
    print('Split into train and validation')
    data_train, data_val = train_test_split(train, test_size=PARAMS[exp_name]['TEST_SIZE'], random_state=PARAMS[exp_name]['SEED'])
    data_train.index     = range(len(data_train))
    data_val.index       = range(len(data_val))

    return data_train, data_val

def iterate_batches(matrix, labels, batch_size, predict_mode='train'):
    indices = np.arange(len(matrix))

    if predict_mode == 'train':
        np.random.shuffle(indices)

    for start in range(0, len(matrix), batch_size):
        end = min(start + batch_size, len(matrix))
        batch_indices = indices[start: end]
        X = matrix[batch_indices]

        if predict_mode != 'train': yield X
        else: yield X, labels[batch_indices]

