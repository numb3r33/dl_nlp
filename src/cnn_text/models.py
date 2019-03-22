import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TConvolution(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(TConvolution, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cin         = 1
        self.Cout        = 1

        self.embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.embedding.weight = nn.Parameter(pre_trained_embeddings)

        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(3, self.hidden_dim))
        self.conv_layer2 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(4, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(2, self.num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        # create a matrix of shape ( N, Cin, max_len, embedding_dim)
        emb = emb.unsqueeze(1)

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(emb)
        out1 = self.relu(out1)
        out1 = out1.squeeze(3)

        out2  = self.conv_layer2(emb)
        out2  = self.relu(out2)
        out2  = out2.squeeze(3)

        # global max pool
        out1, _ = torch.max(out1, dim=-1)
        out2, _ = torch.max(out2, dim=-1)

        out     = torch.cat((out1, out2), dim=1)

        # fully connected layer
        out     = self.fc(out)

        return out

