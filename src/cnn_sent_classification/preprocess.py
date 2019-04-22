import spacy
from collections import Counter, defaultdict
from exp_config import PAD_TOKEN

class Tokenizer():
    def __init__(self, lang='en'):
        self.tok = spacy.blank(lang, disable=['parser', 'tagger', 'ner'])
    
    def tokenizer(self, t):
        return [x.text for x in self.tok.tokenizer(t)]


class Vocab():
    def __init__(self, itos):
        self.itos = itos
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})
    
    def numericalize(self, t):
        return [self.stoi[w] for w in t]
    
    def fix_len(self, sent_len, numericalized_tokens):
        return [nt[:sent_len] for nt in numericalized_tokens]
    
    def __getstate__(self):
        return {'itos': self.itos}
    
    def textify(self, nums, sep=' '):
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums]) if sep is not None else [self.itos[i] for i in nums]

    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o, c in freq.most_common(max_vocab) if c >= min_freq]
        itos = cls.add_special_symbols(itos)
        return cls(itos)
    
    @classmethod
    def add_special_symbols(cls, itos):
        pad_sym = PAD_TOKEN 
        itos.append(pad_sym)
        return itos
