import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from pathlib import Path


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 2, 3, 1)

        return x


class RCNN(nn.Module):
    def __init__(self, emb_matrix, vocab_size, embed_size, config, pad_idx):
        super(RCNN, self).__init__()
        
        # make sure embedding are non-trainable
        self.embedding        = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(emb_matrix)

        self.n_filters    = config['n_filters']
        self.filter_size  = config['filter_size']
        self.hidden_size  = config['hidden_size']

        self.conv     = nn.Conv1d(in_channels=self.hidden_size * 2 + embed_size, 
                                  out_channels=self.n_filters, kernel_size=self.filter_size)
        
        self.lstm     = nn.LSTM(embed_size, self.hidden_size, num_layers=1)
        self.relu     = nn.ReLU()
        self.tanh     = nn.Tanh()
        self.dropout  = nn.Dropout(config['dropout'])
        self.spatial_dropout = SpatialDropout(config['spatial_dropout'])
        self.fc       = nn.Linear(self.n_filters, 6)
        
    def forward(self, x):
        embed    = self.embedding(x)
        
        # introduce channel
        embed    = embed.unsqueeze(1)
   
        # spatial dropout on channels
        embed    = self.spatial_dropout(embed)

        # squeeze channel
        embed    = embed.squeeze(1)

        # left context
        left_context  = F.pad(embed, (0, 0, 1, 0, 0, 0))[:, :-1, :]
        right_context = F.pad(embed, (0, 0, 0, 1, 0, 0))[:, 1:, :]

        left_context  = left_context.permute(1, 0, 2)
        fwd, _        = self.lstm(left_context)

        right_context_reversed = torch.flip(right_context, [1])
        right_context_reversed = right_context_reversed.permute(1, 0, 2)

        bwd, _ = self.lstm(right_context_reversed)
        bwd    = torch.flip(bwd, [1])

        fwd    = fwd.permute(1, 0, 2)
        bwd    = bwd.permute(1, 0, 2)

        out    = torch.cat((fwd, embed, bwd), dim=2)

        out    = out.permute(0, 2, 1)
        out    = self.conv(out)
        out    = self.tanh(out)

        out    = out.max(dim=2)[0]
        out    = self.fc(out)
        
        return out
