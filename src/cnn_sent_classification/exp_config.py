PAD_TOKEN = 'xxxpad'
UNK_TOKEN = 'xxxunk'
W2V_PATH  = '../../data/processed/word2vec.bin.gz'
FASTTEXT_PATH =  '../../data/processed/crawl-300d-2M.vec'
TEXT_COL   = 'comment_text'
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

config = {
    'simple_cnn': {
        'lr': 1e-3,
        'sent_len': 200,
        'emb_type': 'fasttext',
        'dropout': 0.3,
        'spatial_dropout': 0.3,
        'max_vocab' : 100000,
        'min_freq'  : 3,
        'embed_size' : 300,
        'valid_pct'  : 0.2,
        'n_filters': [100, 100, 100],
        'filter_size': [3, 4, 5],
        'ncpus': 8,
        'batch_size': 128,
        'N_EPOCHS': 2
         },
    'rcnn': {
        'lr': 1e-3,
        'sent_len': 200,
        'emb_type': 'fasttext',
        'dropout': 0.3,
        'spatial_dropout': 0.3,
        'max_vocab' : 100000,
        'min_freq'  : 3,
        'embed_size' : 300,
        'valid_pct'  : 0.2,
        'n_filters': 32,
        'hidden_size': 300,
        'filter_size': 1,
        'ncpus': 8,
        'batch_size': 128,
        'N_EPOCHS': 2
         }

    }
