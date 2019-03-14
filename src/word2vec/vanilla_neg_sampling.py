import numpy as np
import pandas as pd
import argparse

from utils import *
from train import *

def main(data_path, vis_fn):
    text = read_data(data_path)
    tokenized_text = tokenize(text)
    vocabulary     = build_vocabulary(tokenized_text)

    word2index, index2word  = numericalization(vocabulary)
    contexts                = build_contexts(tokenized_text, config.WINDOW_SIZE, word2index)
    V                       = len(word2index.keys())
    p_w                     = get_pw(word2index)

    print('sample of some words: {}'.format(np.random.choice(list(word2index.keys()), 5)))

    model = train_w2v_vanilla_neg_sampling(contexts, V, num_skips=config.NUM_SKIPS, batch_size=config.BATCH_SIZE, word2index=word2index, p_w=p_w)
    embeddings = get_embedding(model[0])
    print(most_similar(embeddings, index2word, word2index, config.WORD))

    visualize_embeddings(embeddings, index2word, config.SAMPLE, vis_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Processing with vanilla negative sampling.')

    parser.add_argument('-data_path', help='Path to the data')
    parser.add_argument('-vis_fn', help='Filename of the 2d-projection')

    args = parser.parse_args()
    data_path = args.data_path
    vis_fn    = args.vis_fn
    
    main(data_path, vis_fn)
