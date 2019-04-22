PAD_TOKEN = 'xxxpad'
W2V_PATH  = '../../data/processed/word2vec.bin.gz'
FASTTEXT_PATH =  '../../data/processed/crawl-300d-2M.vec'
TEXT_COL   = 'comment_text'
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

config = {
    'lr': 1e-2,
    'sent_len': 200,
    'emb_type': 'glove',
    #'emb_type': 'fasttext',
    'dropout': 0.1,
    'spatial_dropout': 0.4,
    'max_vocab' : 100000,
    'min_freq'  : 3,
    'embed_size' : 300,
    'valid_pct'  : 0.2,
    'n_filters': [32, 32, 32, 32],
    'filter_size': [1, 2, 3, 5],
    'ncpus': 8,
    'batch_size': 512,
    'N_EPOCHS': 2
    }
