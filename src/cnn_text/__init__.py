import pandas as pd
import numpy as np
import os

import argparse

from utils import *
from train import *

SAMPLE_BASEPATH = '../../data/jigsaw_toxic/processed/'
BASEPATH        = '../../data/jigsaw_toxic/raw/'
SUB_PATH        = '../../submissions/'

def main(exp_name, run_mode):
    if run_mode == 'train':
        train       = read_csv(os.path.join(BASEPATH, 'train_sample.csv'))
        TARGET_COLS = get_target_cols()

        train       = create_new_target(train)
        train_tokenized_comments   = get_tokenized_comments(train.comment_text)
        train['tokenized_comments'] = get_tokenized_comments_string(train.comment_text)
        word2vec                   = define_word2vec(train_tokenized_comments, exp_name)
        words                      = get_word2vec_vocab(word2vec)
        word_vectors               = get_word_vectors(word2vec, words)
        emb_mean, emb_std          = get_mean_std_embedding(word_vectors)

        embedding_matrix, UNK_IX, PAD_IX = create_embedding(word_vectors, words, word2vec, emb_mean, emb_std, exp_name)
        token_to_id                = get_token_to_id(words, word2vec, UNK_IX, PAD_IX)
        data_train, data_val       = get_train_test_splits(train, exp_name)
        model, preds               = train_and_evaluate(embedding_matrix, token_to_id, exp_name, data_train, data_val, TARGET_COLS, UNK_IX, PAD_IX, run_mode)

        # read the train, dev and test datasets.
        # read train, test datasets and cv indices for different folds.
        # read train and test datasets only ( don't worry for holdout )
        # clean all the comments.
        # tokenized all the comments.
        # create a new target label based on the given labels.
        # create vocabulary
        # create token to id mapping
        # convert text into padded matrix of tokens.
        # create a method for batch preparation
        # define model
        # define training loop
        # define method that would save model to disk
        # define method that would load model from disk
        # define predict method
        # define method to save OOF predictions
    else:
        train       = read_csv(os.path.join(BASEPATH, 'train.csv'))
        test        = read_csv(os.path.join(BASEPATH, 'test.csv'))
        test_labels = read_csv(os.path.join(BASEPATH, 'test_labels.csv'))

        TARGET_COLS = get_target_cols()

        train       = create_new_target(train)
        train_tokenized_comments    = get_tokenized_comments(train.comment_text)
        train['tokenized_comments'] = get_tokenized_comments_string(train.comment_text)
        test['tokenized_comments']  = get_tokenized_comments_string(test.comment_text)

        word2vec                   = define_word2vec(train_tokenized_comments, exp_name)
        words                      = get_word2vec_vocab(word2vec)
        word_vectors               = get_word_vectors(word2vec, words)
        emb_mean, emb_std          = get_mean_std_embedding(word_vectors)

        embedding_matrix, UNK_IX, PAD_IX = create_embedding(word_vectors, words, word2vec, emb_mean, emb_std, exp_name)
        token_to_id                = get_token_to_id(words, word2vec, UNK_IX, PAD_IX)
        model, preds               = train_and_evaluate(embedding_matrix, token_to_id, exp_name, train, test, TARGET_COLS, UNK_IX, PAD_IX, run_mode)

        prepare_submission(test_labels, preds, os.path.join(SUB_PATH, 'cnn_text.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN for Text Classification ( Kaggle Jigsaw Toxic Comments Classification )')

    parser.add_argument('-run_mode', help='Run the model in train or eval mode')
    parser.add_argument('-exp_name', help='Name of the Experiment')

    args = parser.parse_args()

    run_mode = args.run_mode
    exp_name = args.exp_name

    main(exp_name, run_mode)
