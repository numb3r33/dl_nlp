import pandas as pd
import numpy as np
import torch

from pathlib import Path
from preprocess import Tokenizer, Vocab
from utils import check_labels, load_w2v_embedding, load_fasttext_embedding, pad_collate
from torch.utils.data import Dataset, DataLoader
from functools import partial
from exp_config import TEXT_COL, LABEL_COLS, PAD_TOKEN


class TextLMData():  
    def __init__(self, 
                 path,
                 csv,
                 test_csv,
                 text_col, 
                 label_cols, 
                 max_vocab,
                 min_freq,
                 valid_pct=0.2):
        
        self.path       = path
        self.csv        = csv
        self.test_csv   = test_csv
        self.text_cols  = text_col
        self.label_cols = label_cols
        self.valid_pct  = valid_pct
        self.max_vocab  = max_vocab
        self.min_freq   = min_freq
        
        self.df      = pd.read_csv(Path(self.path)/self.csv)
        if self.test_csv is not None: self.test_df = pd.read_csv(Path(self.path)/self.test_csv) 
        self.cut     = int(valid_pct * len(self.df)) + 1
        
    def process(self):
        tok = Tokenizer()
        
        # consider entire corpus as text ( train + test text columns )
        if self.test_csv:
            text = list(self.df.loc[:, self.text_cols].values) + list(self.test_df.loc[:, self.text_cols])
        else:
            text = list(self.df.loc[:, self.text_cols].values)
        
        self.tokens  = [tok.tokenizer(x) for x in text]
        self.vocab   = Vocab.create(self.tokens, self.max_vocab, self.min_freq)
        
        self.ntokens = [self.vocab.numericalize(t) for t in self.tokens]
        
        # only full training
        if self.valid_pct == 0 and self.test_csv is None:
            self.trn_ds  = (self.ntokens, self.df.loc[:, self.label_cols].values)
            self.vld_ds  = ([], [])
            self.test_ds = ([], [])
        
        # holdout
        elif self.valid_pct > 0 and self.test_csv is None:
            self.trn_ds  = (self.ntokens[self.cut:], self.df.loc[:, self.label_cols].values[self.cut:])
            self.vld_ds  = (self.ntokens[:self.cut], self.df.loc[:, self.label_cols].values[:self.cut])
            self.tst_ds  = ([], [])
        
        # holdout and test prediction
        elif self.valid_pct > 0 and self.test_csv is not None:
            self.trn_tokens  = self.ntokens[:len(self.df)]
            self.tst_ds      = (self.ntokens[len(self.df):], [])
            
            trn_tokens  = self.trn_tokens[self.cut:]
            vld_tokens  = self.trn_tokens[:self.cut]
            
            self.trn_ds = (trn_tokens, self.df.loc[:, self.label_cols].values[self.cut:])
            self.vld_ds = (vld_tokens, self.df.loc[:, self.label_cols].values[:self.cut])
        
        # full training and test prediction
        else:
            self.trn_ds  = (self.ntokens[:len(self.df)], self.df.loc[:, self.label_cols].values)
            self.vld_ds  = ([], [])
            self.tst_ds  = (self.ntokens[len(self.df):], [])
            
        return self.vocab, self.trn_ds, self.vld_ds, self.tst_ds
    
    def fill_emb_matrix(self,  emb_type, embed_size):
        emb_matrix = np.random.random(size=(len(self.vocab.itos), embed_size))
        
        if emb_type == 'w2v':
            emb_matrix = load_w2v_embedding(emb_matrix, self.vocab, embed_size)
        elif emb_type == 'fasttext':
            emb_matrix = load_fasttext_embedding(emb_matrix, self.vocab, embed_size)
            
        return emb_matrix


class TextClassData(Dataset):
    def __init__(self, vocab, ds):
        self.vocab       = vocab
        self.ds, self.y  = ds
                            
    def __len__(self):
        return len(self.ds)
                            
    def __getitem__(self, index):
        x = torch.LongTensor(self.ds[index])
        y = None
        if len(self.y) > 0: y = torch.FloatTensor(self.y[index])
        
        return x, y


def make_dataset(config):

    path       = Path(config['path'])
    csv        = config['csv']
    text_col   = TEXT_COL 
    label_cols = LABEL_COLS
    test_csv   = config['test_csv']
    max_vocab  = config['max_vocab']
    min_freq   = config['min_freq']
    embed_size = config['embed_size']
    emb_type   = config['emb_type']
    valid_pct  = config['valid_pct']

    lm         = TextLMData(path,
                            csv,
                            test_csv,
                            text_col,
                            label_cols,
                            max_vocab,
                            min_freq,
                            valid_pct=valid_pct)

    vocab, trn_ds, vld_ds, tst_ds = lm.process()
    emb_matrix                    = lm.fill_emb_matrix(emb_type, embed_size)
    
    return vocab, trn_ds, vld_ds, tst_ds, emb_matrix


def make_iterator(config, vocab, trn_ds, vld_ds, tst_ds):
    sent_len   = config['sent_len']
    collate_fn = partial(pad_collate, pad_idx=vocab.stoi[PAD_TOKEN], sent_len=sent_len)
    
    trn_ds     = TextClassData(vocab, trn_ds)
    if len(vld_ds) > 0: vld_ds = TextClassData(vocab, vld_ds)
    if len(tst_ds) > 0: tst_ds = TextClassData(vocab, tst_ds)

    trn_dl = DataLoader(trn_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config['ncpus'])
    vld_dl, tst_dl = None, None

    if len(vld_ds) > 0: vld_dl = DataLoader(vld_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config['ncpus'])
    if len(tst_ds) > 0: tst_dl = DataLoader(tst_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config['ncpus'])
    
    return trn_dl, vld_dl, tst_dl
