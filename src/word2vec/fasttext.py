import numpy as np
import pandas as pd
import argparse
import os

from utils import *
from train import *

from sklearn.externals import joblib

def main(data_path, vis_fn, model_name):
    text           = read_data(data_path)
    tokenized_text = tokenize(text)
    vocabulary     = build_vocabulary(tokenized_text)

    word2index, index2word  = numericalization(vocabulary)
    ngram_hash, hash_ngram, word_ngram_index, word_ngram = generate_character_ngram(word2index)

    contexts                = build_contexts(tokenized_text, config.WINDOW_SIZE, word2index)
    V                       = len(word2index.keys())
    p_w                     = get_pw(word2index)

    print('sample of some words: {}'.format(np.random.choice(list(word2index.keys()), 5)))

    if not os.path.exists(os.path.join('../../models/', f'{model_name}.pkl')):
        print('Training word embeddings ...')
        model = train_fasttext(contexts, V, num_skips=config.NUM_SKIPS, batch_size=config.FT_BATCH_SIZE, ngram_hash=ngram_hash, word_ngram_index=word_ngram_index, word2index=word2index, p_w=p_w)

        joblib.dump(model, os.path.join('../../models/', f'{model_name}.pkl'))
    else:
        print('Loading model from disk ...')
        model = joblib.load(os.path.join('../../models/', f'{model_name}.pkl'))

    embeddings = get_embedding(model.embedding)
    projection = model.linear.weight.cpu().detach().numpy()

    print(most_similar_ft(embeddings, projection, ngram_hash,  hash_ngram, word_ngram_index, word2index, index2word, config.WORD))
    visualize_embeddings(embeddings, index2word, config.SAMPLE, vis_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate word embeddings using Fasttext.')

    parser.add_argument('-data_path', help='Path to the data')
    parser.add_argument('-vis_fn', help='Filename of the 2d-projection')
    parser.add_argument('-model_name', help='Filename of the model')

    args       = parser.parse_args()
    data_path  = args.data_path
    vis_fn     = args.vis_fn
    model_name = args.model_name

    main(data_path, vis_fn, model_name)
