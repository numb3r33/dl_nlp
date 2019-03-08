import numpy as np
import pandas as pd
import argparse

from utils import *

def main(data_path):
    text = read_data(data_path)
    tokenized_text = tokenize(text)
    vocabulary     = build_vocabulary(tokenized_text)

    word2index, index2word  = numericalization(vocabulary)
    contexts                = build_contexts(tokenized_text, config.WINDOW_SIZE, word2index)

    print('Length of the vocabulary: {}'.format(len(vocabulary.keys())))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Processing')

    parser.add_argument('-data_path', help='Path to the data')
    args = parser.parse_args()
    data_path = args.data_path
    
    main(data_path)
