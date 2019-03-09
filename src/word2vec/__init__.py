import numpy as np
import pandas as pd
import argparse

from utils import *
from train import *

def main(data_path):
    text = read_data(data_path)
    tokenized_text = tokenize(text)
    vocabulary     = build_vocabulary(tokenized_text)

    word2index, index2word  = numericalization(vocabulary)
    contexts                = build_contexts(tokenized_text, config.WINDOW_SIZE, word2index)
    V                       = len(word2index.keys())
    
    print('sample of some words: {}'.format(np.random.choice(list(word2index.keys()), 5)))
    model = train_w2v(contexts, V, num_skips=config.NUM_SKIPS, batch_size=config.BATCH_SIZE)
    embeddings = get_embedding(model)
    print(most_similar(embeddings, index2word, word2index, config.WORD))

    visualize_embeddings(embeddings, index2word, config.SAMPLE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Processing')

    parser.add_argument('-data_path', help='Path to the data')
    args = parser.parse_args()
    data_path = args.data_path
    
    main(data_path)
