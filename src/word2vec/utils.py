from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict

import random
import numpy as np
import config
import math

from sklearn.metrics.pairwise import cosine_similarity

def read_data(path):
    text = ''

    with open(path, 'r') as infile:
        for row in infile.readlines():
            text += row

    return text[:100000]


def clean(text):
    pass

def tokenize(text):
    tokenized_text = word_tokenize(text)
    return tokenized_text

def build_vocabulary(tokenized_text):
    MIN_COUNT = config.MIN_COUNT

    words_counter = Counter(token for token in tokenized_text)
    vocabulary    = {}

    for word, count in words_counter.most_common():
        if count >= MIN_COUNT:
            vocabulary[word] = count

    return vocabulary

def numericalization(vocabulary):
    word2index = {'<unk>': 0}

    for word, count in vocabulary.items():
        word2index[word] = len(word2index)

    index2word = [word for word, _ in sorted(word2index.items(), key=lambda x: x[1])]
    print('Most freq words: {}'.format(index2word[1:10]))

    return word2index, index2word

def build_contexts(tokenized_text, window_size, word2index):
    contexts = []

    for i in range(len(tokenized_text)):
        central_word = tokenized_text[i]
        context = [tokenized_text[i + delta] for delta in range(-window_size, window_size + 1)\
                                           if delta != 0 and i + delta >= 0 and i + delta < len(tokenized_text)]

        contexts.append((central_word, context))

    print('Example of some of the context values\n')
    print(contexts[:3])

    contexts = [(word2index.get(central_word, 0), [word2index.get(word, 0) for word in context])\
                 for central_word, context in contexts]

    return contexts


def make_skip_gram_batchs_iter(contexts, window_size, num_skips, batch_size):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window_size

    central_words = [word for word, context in contexts if len(context) == 2 * window_size and word != 0]
    contexts = [context for word, context in contexts if len(context) == 2 * window_size and word != 0]

    batch_size   = int(batch_size / num_skips)
    batchs_count = int(math.ceil(len(contexts) / batch_size))

    print('Initializing batchs generator with {} batchs per epoch'.format(batchs_count))
    indices = np.arange(len(contexts))
    np.random.shuffle(indices)

    for i in range(batchs_count):
        batch_begin, batch_end = i * batch_size, min((i + 1) * batch_size, len(contexts))
        batch_indices = indices[batch_begin: batch_end]

        batch_data, batch_labels = [], []

        for data_ind in batch_indices:
            central_word, context = central_words[data_ind], contexts[data_ind]
            words_to_use = random.sample(context, num_skips)

            batch_data.extend([central_word] * num_skips)
            batch_labels.extend(words_to_use)

        yield batch_data, batch_labels


def get_embedding(embedding):
    return embedding.weight.cpu().detach().numpy()

def most_similar(embeddings, index2word, word2index, word):
    word_emb = embeddings[word2index[word]]

    sim = cosine_similarity([word_emb], embeddings)[0]
    top10 = np.argsort(sim)[-10:]

    return [index2word[index] for index in reversed(top10)]

def get_token_ngram(token):
    ngrams = []
    ngram_range = config.NGRAM_RANGE

    if token == '<unk>':
        ngrams.append(token)

    token = '<' + token + '>'

    for i in range(len(token) - ngram_range + 1):
        ngrams.append(token[i:i+ngram_range])

    ngrams.append(token)

    return ngrams

def get_query_word_embedding(word_embeddings, embeddings, word2index, ngram_hash, word):
    if word in word2index:
        return word_embeddings[word2index[word]]
    else:
        print()
        print('Word not found in training')
        ngrams   = get_token_ngram(word)
        q_emb    = np.zeros(shape=(config.EMBEDDING_SIZE, ))
        count_ng = 0

        for ng in ngrams:
            if ng in ngram_hash:
                q_emb += embeddings[ngram_hash[ng]]
                count_ng += 1

        #q_emb /= count_ng

        return q_emb

def most_similar_ft(embeddings, projection, ngram_hash, hash_ngram, word_ngram_index, word2index, index2word, word):
    # create word embeddings out of character embeddings
    word_embeddings = np.zeros(shape=(len(word_ngram_index.keys()), config.EMBEDDING_SIZE))

    for index, k in enumerate(word_ngram_index.keys()):
        chars                     = word_ngram_index[k]
        #word_embeddings[index, :] = np.sum(embeddings[chars], axis=0) / len(chars)
        word_embeddings[index, :] = np.sum(embeddings[chars], axis=0)

    word_emb     = get_query_word_embedding(word_embeddings, embeddings, word2index, ngram_hash, word)
    similarities = cosine_similarity([word_emb], word_embeddings)[0]
    top10        = np.argsort(similarities)[-10:]

    return [index2word[index] for index in reversed(top10)]

def generate_character_ngram(word2index):
    ngram_range = config.NGRAM_RANGE
    word_ngram  = defaultdict(list)

    ngrams      = []

    for token in word2index.keys():
        ngram = []

        if token == '<unk>':
            ngrams.append(token)
            ngram.append(token)
            word_ngram[token].extend(list(set(ngram)))
            continue

        token = '<' + token + '>'

        for i in range(len(token) - ngram_range + 1):
            ngrams.append(token[i:i+ngram_range])
            ngram.append(token[i:i+ngram_range])

        ngram.append(token)
        ngrams.append(token)

        word_ngram[token[1:-1]].extend(list(set(ngram)))

    ngrams = list(set(ngrams))

    ngram_hash = {}

    for ngram in ngrams:
        ngram_hash[ngram] = len(ngram_hash) + 1

    print('Number of charater n-grams: {}'.format(len(ngram_hash)))

    hash_ngram = {}

    for  k,v in ngram_hash.items():
        hash_ngram[v] = k

    word_ngram_index = {}

    for k,v in word_ngram.items():
        word_ngram_index[word2index[k]] = [ngram_hash[x] for x in v]

    return ngram_hash, hash_ngram, word_ngram_index, word_ngram

def pad_to_dense_3D(M, num_skips):
    maxlen = max(len(s) for r in M for s in r)
    Z      = np.zeros((len(M), num_skips, maxlen))

    for enu, row in enumerate(M):
        for skip in range(num_skips):
            Z[enu, skip, :len(row[skip])] += row[skip]

    return Z.astype(np.int)

def get_pw(word2index):
    factor = 0

    for k, v in word2index.items():
        factor += (v ** config.UNIGRAM_POWER)

    factor = 1 / factor
    p_w    = {k: (v ** config.UNIGRAM_POWER) * factor for k,v in word2index.items()}

    return p_w

def get_neg_word_indices_ft(batch_len, context_window, word2index, p_w):
    NUM_WORDS = config.NUM_NEG_WORDS
    neg_word_indices = []

    for i in range(batch_len):
        nws = []
        for j in range(context_window):
            neg_words = np.random.choice([k for k in word2index.keys()], size=NUM_WORDS, replace=False, p=[v for v in p_w.values()])
            nws.append(np.array([word2index[w] for w in neg_words]))

        neg_word_indices.append(nws)

    return neg_word_indices

def get_pos_word_indices(batch_len, context_window, labels):
    pos_words_indices = []

    for i in range(batch_len):
        pws = []

        for j in range(context_window):
            pws.append([labels[i][j].cpu().item()])

        pos_words_indices.append(pws)

    return pos_words_indices



def make_skip_gram_batchs_iter_with_context(contexts, window_size, num_skips, batch_size):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window_size

    central_words = [word for word, context in contexts if len(context) == 2 * window_size and word != 0]
    contexts = [context for word, context in contexts if len(context) == 2 * window_size and word != 0]

    batch_size   = int(batch_size / num_skips)
    batchs_count = int(math.ceil(len(contexts) / batch_size))

    print('Initializing batchs generator with {} batchs per epoch'.format(batchs_count))
    indices = np.arange(len(contexts))
    np.random.shuffle(indices)

    for i in range(batchs_count):
        batch_begin, batch_end = i * batch_size, min((i + 1) * batch_size, len(contexts))
        batch_indices = indices[batch_begin: batch_end]

        batch_data, batch_labels = [], []

        for data_ind in batch_indices:
            central_word, context = central_words[data_ind], contexts[data_ind]
            words_to_use = random.sample(context, num_skips)

            batch_data.extend([words_to_use])
            batch_labels.extend([central_word])

        yield batch_data, batch_labels

def make_skip_gram_batchs_iter_for_fasttext(contexts, window_size, num_skips, batch_size, word_ngram_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window_size

    central_words = [word for word, context in contexts if len(context) == 2 * window_size and word != 0]
    contexts = [context for word, context in contexts if len(context) == 2 * window_size and word != 0]

    batch_size   = int(batch_size / num_skips)
    batchs_count = int(math.ceil(len(contexts) / batch_size))

    print('Initializing batchs generator with {} batchs per epoch'.format(batchs_count))
    indices = np.arange(len(contexts))
    np.random.shuffle(indices)

    for i in range(batchs_count):
        batch_begin, batch_end = i * batch_size, min((i + 1) * batch_size, len(contexts))
        batch_indices = indices[batch_begin: batch_end]

        batch_data, batch_labels = [], []

        for data_ind in batch_indices:
            central_word, context = central_words[data_ind], contexts[data_ind]
            words_to_use = random.sample(context, num_skips)

            batch_data.extend([[word_ngram_index[central_word]] * num_skips])
            batch_labels.extend([words_to_use])

        batch_data = pad_to_dense_3D(batch_data, num_skips)
        yield batch_data, batch_labels
