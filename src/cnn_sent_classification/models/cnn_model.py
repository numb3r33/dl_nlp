import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from pathlib import Path


class CNNModel(nn.Module):
    def __init__(self, emb_matrix, vocab_size, embed_size, config):
        super(CNNModel, self).__init__()
        
        self.embedding        = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(emb_matrix)
        self.embedding.weight.requires_grad = False
        
        self.ns_embedding = nn.Embedding(vocab_size, embed_size)
        self.ns_embedding.weight = nn.Parameter(emb_matrix)
        

        self.n_filters    = config['n_filters']
        self.filter_sizes = config['filter_size']
        self.convs        = nn.ModuleList([
                                    nn.Conv2d(in_channels = 2, 
                                              out_channels = self.n_filters[i], 
                                              kernel_size = (fs, embed_size)) 
                                    for i, fs in enumerate(self.filter_sizes)
                                    ])
        
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(config['dropout'])
        self.spatial_dropout = nn.Dropout2d(config['spatial_dropout'])
        self.fc       = nn.Linear(np.sum(self.n_filters), 6)
        
    def forward(self, x):
        embed    = self.embedding(x)
        ns_embed = self.ns_embedding(x) 
        
        # introduce channel
        embed    = embed.unsqueeze(1)
        ns_embed = ns_embed.unsqueeze(1)
        
        embed    = torch.cat((embed, ns_embed), dim=1)
        
        # spatial dropout on channels
        embed    = self.spatial_dropout(embed)
        
        conved = [self.relu(conv(embed)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    
        # pass to dropout
        out    = self.dropout(torch.cat(pooled, dim=1))
        
        # pass to fully connected layer
        fc     = self.fc(out)
        
        return fc
