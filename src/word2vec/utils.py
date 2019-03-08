from nltk.tokenize import word_tokenize
from collections import Counter

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


def get_embedding(model):
    return model[0].weight.cpu().detach().numpy()

def most_similar(embeddings, index2word, word2index, word):
    word_emb = embeddings[word2index[word]]

    sim = cosine_similarity([word_emb], embeddings)[0]
    top10 = np.argsort(sim)[-10:]

    return [index2word[index] for index in reversed(top10)]
