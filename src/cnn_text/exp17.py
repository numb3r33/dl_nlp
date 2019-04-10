import pandas as pd
import numpy as np
import os

import argparse

from utils import *
from train import *
from losses import *
from models import *

SAMPLE_BASEPATH = '../../data/processed/'
BASEPATH        = '../../data/raw/'
SUB_PATH        = '../submissions/'
MODEL_FP        = '../../models/'
SUB_FN          = 'exp17_cnn.csv'


def main(exp_name, run_mode, is_sample):

    if run_mode == 'train':

        if is_sample:
            train = read_csv(os.path.join(SAMPLE_BASEPATH, 'train_sample.csv'))
        else:
            train = read_csv(os.path.join(BASEPATH, 'train.csv'))

        TARGET_COLS = get_target_cols()

        train_tokenized_comments    = get_tokenized_comments(train.comment_text)
        train['tokenized_comments'] = get_tokenized_comments_string(train.comment_text)
        words                       = get_vocab(train_tokenized_comments, exp_name)


        embedding_matrix, UNK, PAD, UNK_IX, PAD_IX = load_fasttext_embedding_matrix(words)
        token_to_id                = get_token_to_id(words, UNK, PAD, UNK_IX, PAD_IX)
        print('Vocabulary |V|: {}'.format(len(token_to_id)))

        data_train, data_val       = get_train_test_splits(train, exp_name)

        model     = get_exp17_model(embedding_matrix, token_to_id, exp_name, PAD_IX)
        criterion = get_exp2_criterion()
        optimizer = get_exp2_optimizer(model, exp_name)


        model, preds = train_and_evaluate(model, criterion, optimizer, embedding_matrix, token_to_id, exp_name, data_train, data_val, TARGET_COLS, UNK_IX, PAD_IX, run_mode)

    else:
        train       = read_csv(os.path.join(BASEPATH, 'train.csv'))
        test        = read_csv(os.path.join(BASEPATH, 'test.csv'))
        test_labels = read_csv(os.path.join(BASEPATH, 'test_labels.csv'))

        TARGET_COLS = get_target_cols()

        train_tokenized_comments    = get_tokenized_comments(train.comment_text)
        train['tokenized_comments'] = get_tokenized_comments_string(train.comment_text)
        test['tokenized_comments']  = get_tokenized_comments_string(test.comment_text)

        words                       = get_vocab(train_tokenized_comments, exp_name)

        embedding_matrix, UNK, PAD, UNK_IX, PAD_IX = load_fasttext_embedding_matrix(words)
        token_to_id                = get_token_to_id(words, UNK, PAD, UNK_IX, PAD_IX)

        model     = get_exp17_model(embedding_matrix, token_to_id, exp_name, PAD_IX)
        criterion = get_exp2_criterion()
        optimizer = get_exp2_optimizer(model, exp_name)


        model, preds = train_and_evaluate(model, criterion, optimizer, embedding_matrix, token_to_id, exp_name, train, test, TARGET_COLS, UNK_IX, PAD_IX, run_mode)

        prepare_submission(test_labels, preds, os.path.join(SUB_PATH, SUB_FN))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN for Text Classification ( Kaggle Jigsaw Toxic Comments Classification )')

    parser.add_argument('-run_mode', type=str, help='Run the model in train or eval mode')
    parser.add_argument('-exp_name', type=str, help='Name of the Experiment')
    parser.add_argument('-is_sample', type=bool, help='Do you want to run the experiment on the sample?')

    args = parser.parse_args()

    run_mode  = args.run_mode
    exp_name  = args.exp_name
    is_sample = args.is_sample

    main(exp_name, run_mode, is_sample)
