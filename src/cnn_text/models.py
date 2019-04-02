import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import *

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


class Experiment2(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment2, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cin         = 1
        self.Cout        = 1

        self.embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.embedding.weight.requires_grad = False


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


class Experiment3(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment3, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cin         = 32
        self.Cout        = 1

        self.embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.embedding.weight.requires_grad = False


        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(3, self.hidden_dim))
        self.conv_layer2 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(4, self.hidden_dim))
        self.conv_layer3 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(1, self.hidden_dim))
        self.conv_layer4 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(5, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(4 * self.Cin, self.num_classes)
        self.dropout     = nn.Dropout(0.1)

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

        out3  = self.conv_layer3(emb)
        out3  = self.relu(out3)
        out3  = out3.squeeze(3)

        out4  = self.conv_layer4(emb)
        out4  = self.relu(out4)
        out4  = out4.squeeze(3)

        # global max pool
        out1, _ = torch.max(out1, dim=-1)
        out2, _ = torch.max(out2, dim=-1)
        out3, _ = torch.max(out3, dim=-1)
        out4, _ = torch.max(out4, dim=-1)

        # concantenate all the outputs together
        out     = torch.cat((out1, out2, out3, out4), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment4(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment4, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cin         = 100
        self.Cout        = 2

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # second embedding layer that is not-static ( both of them are pre-trained embeddings )
        self.non_static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.non_static_embedding.weight = nn.Parameter(pre_trained_embeddings)


        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(3, self.hidden_dim))
        self.conv_layer2 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(4, self.hidden_dim))
        self.conv_layer3 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(2, self.hidden_dim))
        self.conv_layer4 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(5, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(4 * self.Cin, self.num_classes)
        self.dropout     = nn.Dropout(0.5)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        # create a matrix of shape ( N, Cin, max_len, embedding_dim)
        static_emb = static_emb.unsqueeze(1)

        non_static_emb = self.non_static_embedding(x)
        non_static_emb = non_static_emb.unsqueeze(1)

        # concatenate static and non-static layers
        emb = torch.cat((static_emb, non_static_emb), dim=1)

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(emb)
        out1 = self.relu(out1)
        out1 = out1.squeeze(3)

        out2  = self.conv_layer2(emb)
        out2  = self.relu(out2)
        out2  = out2.squeeze(3)

        out3  = self.conv_layer3(emb)
        out3  = self.relu(out3)
        out3  = out3.squeeze(3)

        out4  = self.conv_layer4(emb)
        out4  = self.relu(out4)
        out4  = out4.squeeze(3)

        # global max pool
        out1, _ = torch.max(out1, dim=-1)
        out2, _ = torch.max(out2, dim=-1)
        out3, _ = torch.max(out3, dim=-1)
        out4, _ = torch.max(out4, dim=-1)

        # concantenate all the outputs together
        out     = torch.cat((out1, out2, out3, out4), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment5(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment5, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cin         = 32
        self.Cout        = 2

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # second embedding layer that is not-static ( both of them are pre-trained embeddings )
        self.non_static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.non_static_embedding.weight = nn.Parameter(pre_trained_embeddings)


        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(3, self.hidden_dim))
        self.conv_layer2 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(4, self.hidden_dim))
        self.conv_layer3 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(2, self.hidden_dim))
        self.conv_layer4 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(5, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(4 * self.Cin, self.num_classes)
        self.dropout     = nn.Dropout(0.3)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        # create a matrix of shape ( N, Cin, max_len, embedding_dim)
        static_emb = static_emb.unsqueeze(1)

        non_static_emb = self.non_static_embedding(x)
        non_static_emb = non_static_emb.unsqueeze(1)

        # concatenate static and non-static layers
        emb = torch.cat((static_emb, non_static_emb), dim=1)

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(emb)
        out1 = self.relu(out1)
        out1 = out1.squeeze(3)

        out2  = self.conv_layer2(emb)
        out2  = self.relu(out2)
        out2  = out2.squeeze(3)

        out3  = self.conv_layer3(emb)
        out3  = self.relu(out3)
        out3  = out3.squeeze(3)

        out4  = self.conv_layer4(emb)
        out4  = self.relu(out4)
        out4  = out4.squeeze(3)

        # global max pool
        out1, _ = torch.max(out1, dim=-1)
        out2, _ = torch.max(out2, dim=-1)
        out3, _ = torch.max(out3, dim=-1)
        out4, _ = torch.max(out4, dim=-1)

        # concantenate all the outputs together
        out     = torch.cat((out1, out2, out3, out4), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out

class Experiment6(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment6, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cin         = 32
        self.Cout        = 1

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # second embedding layer that is not-static ( both of them are pre-trained embeddings )
        self.non_static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.non_static_embedding.weight = nn.Parameter(pre_trained_embeddings)


        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(3, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(1 * self.Cin, self.num_classes)
        self.dropout     = nn.Dropout(0.4)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        # create a matrix of shape ( N, Cin, max_len, embedding_dim)
        static_emb = static_emb.unsqueeze(1)

        #non_static_emb = self.non_static_embedding(x)
        #non_static_emb = non_static_emb.unsqueeze(1)

        # concatenate static and non-static layers
        #emb = torch.cat((static_emb, non_static_emb), dim=1)
        emb = static_emb

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(emb)
        out1 = self.relu(out1)
        out1 = out1.squeeze(3)

        # global max pool
        out_max_1, _ = torch.max(out1, dim=-1)

        # concantenate all the outputs together
        out     = out_max_1

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment7(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment7, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cins        = [100, 200, 200]
        self.Cout        = 2

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # second embedding layer that is not-static ( both of them are pre-trained embeddings )
        self.non_static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.non_static_embedding.weight = nn.Parameter(pre_trained_embeddings)


        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cins[0], kernel_size=(4, self.hidden_dim))
        self.conv_layer2 = nn.Conv2d(self.Cout, self.Cins[1], kernel_size=(5, self.hidden_dim))
        self.conv_layer3 = nn.Conv2d(self.Cout, self.Cins[2], kernel_size=(6, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(1 * self.Cins[0] + 1 * self.Cins[1] + 1 * self.Cins[2], self.num_classes)
        self.dropout     = nn.Dropout(0.1)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        # create a matrix of shape ( N, Cin, max_len, embedding_dim)
        static_emb = static_emb.unsqueeze(1)

        non_static_emb = self.non_static_embedding(x)
        non_static_emb = non_static_emb.unsqueeze(1)

        # concatenate static and non-static layers
        emb = torch.cat((static_emb, non_static_emb), dim=1)

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(emb)
        out1 = self.relu(out1)
        out1 = out1.squeeze(3)

        out2 = self.conv_layer2(emb)
        out2 = self.relu(out2)
        out2 = out2.squeeze(3)

        out3 = self.conv_layer3(emb)
        out3 = self.relu(out3)
        out3 = out3.squeeze(3)

        # global max pool
        out_max_1, _ = torch.max(out1, dim=-1)
        out_max_2, _ = torch.max(out2, dim=-1)
        out_max_3, _ = torch.max(out3, dim=-1)

        # concantenate all the outputs together
        out     = torch.cat((out_max_1, out_max_2, out_max_3), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out

class Experiment8(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes, PAD_IX):
        super(Experiment8, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.Cins        = [100, 200, 200]
        self.Cout        = 2

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # second embedding layer that is not-static ( both of them are pre-trained embeddings )
        self.non_static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.non_static_embedding.weight = nn.Parameter(pre_trained_embeddings)

        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cins[0], kernel_size=(4, self.hidden_dim))
        self.conv_layer2 = nn.Conv2d(self.Cout, self.Cins[1], kernel_size=(5, self.hidden_dim))
        self.conv_layer3 = nn.Conv2d(self.Cout, self.Cins[2], kernel_size=(6, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(1 * self.Cins[0] + 1 * self.Cins[1] + 1 * self.Cins[2], self.num_classes)
        self.dropout     = nn.Dropout(0.1)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        # create a matrix of shape ( N, Cin, max_len, embedding_dim)
        static_emb = static_emb.unsqueeze(1)

        non_static_emb = self.non_static_embedding(x)
        non_static_emb = non_static_emb.unsqueeze(1)

        # concatenate static and non-static layers
        emb = torch.cat((static_emb, non_static_emb), dim=1)
        #emb = static_emb

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(emb)
        out1 = self.relu(out1)
        out1 = out1.squeeze(3)

        out2 = self.conv_layer2(emb)
        out2 = self.relu(out2)
        out2 = out2.squeeze(3)

        out3 = self.conv_layer3(emb)
        out3 = self.relu(out3)
        out3 = out3.squeeze(3)

        # global max pool
        out_max_1, _ = torch.max(out1, dim=-1)
        out_max_2, _ = torch.max(out2, dim=-1)
        out_max_3, _ = torch.max(out3, dim=-1)

        # concantenate all the outputs together
        out     = torch.cat((out_max_1, out_max_2, out_max_3), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment9(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes):
        super(Experiment9, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.nfms        = [100, 100]
        self.ks          = [4, 5]
        self.Cout        = 1

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False


        self.conv_layer1 = nn.Conv1d(self.hidden_dim, self.nfms[0], self.ks[0])
        self.conv_layer2 = nn.Conv1d(self.hidden_dim, self.nfms[1], self.ks[1])

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(self.nfms[0] + self.nfms[1], self.num_classes)
        self.dropout     = nn.Dropout(0.1)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        static_emb = torch.transpose(static_emb, 1, 2)

        out = static_emb

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(out)
        out1 = self.relu(out1)

        out2 = self.conv_layer2(out)
        out2 = self.relu(out2)

        out1 = torch.transpose(out1, 1, 2)
        out2 = torch.transpose(out2, 1, 2)

        # global max pool
        out1 = out1.max(dim=1)[0]
        out2 = out2.max(dim=1)[0]

        # concantenate all the outputs together
        out     = torch.cat((out1, out2), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment10(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes):
        super(Experiment10, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.nfms        = [150, 150, 150]
        self.ks          = [3, 4, 5]

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False


        self.conv_layer1 = nn.Conv1d(self.hidden_dim, self.nfms[0], self.ks[0])
        self.conv_layer2 = nn.Conv1d(self.hidden_dim, self.nfms[1], self.ks[1])
        self.conv_layer3 = nn.Conv1d(self.hidden_dim, self.nfms[2], self.ks[2])

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(self.nfms[0] + self.nfms[1] +\
                                     self.nfms[2], self.num_classes)
        self.dropout     = nn.Dropout(0.1)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        static_emb = torch.transpose(static_emb, 1, 2)

        out = static_emb

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(out)
        out1 = self.relu(out1)

        out2 = self.conv_layer2(out)
        out2 = self.relu(out2)

        out3 = self.conv_layer3(out)
        out3 = self.relu(out3)

        out1 = torch.transpose(out1, 1, 2)
        out2 = torch.transpose(out2, 1, 2)
        out3 = torch.transpose(out3, 1, 2)

        # global max pool
        out1 = out1.max(dim=1)[0]
        out2 = out2.max(dim=1)[0]
        out3 = out3.max(dim=1)[0]

        # concantenate all the outputs together
        out     = torch.cat((out1, out2, out3), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment12(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes):
        super(Experiment12, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.nfms        = [150, 150, 150, 150]
        self.ks          = [3, 4, 4, 5]
        self.Cout        = 2

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # non-static embedding layer ( specific to the current task )
        self.ns_static_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.ns_static_embedding.weight = nn.Parameter(pre_trained_embeddings)

        self.conv_layer1 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[0], self.ks[0])
        self.conv_layer2 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[1], self.ks[1])
        self.conv_layer3 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[2], self.ks[2])
        self.conv_layer4 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[3], self.ks[3])

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(self.nfms[0] + self.nfms[1] +\
                                     self.nfms[2] + self.nfms[3]
                                     , self.num_classes)

        self.dropout         = nn.Dropout(0.1)
        self.spatial_dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        static_emb = torch.transpose(static_emb, 1, 2)

        ns_static_emb = self.ns_static_embedding(x)
        ns_static_emb = torch.transpose(ns_static_emb, 1, 2)

        out  = torch.cat((static_emb, ns_static_emb), dim = 1)
        #out   = static_emb

        # spatial dropout
        out  = self.spatial_dropout(out)

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(out)
        out1 = self.relu(out1)

        out2 = self.conv_layer2(out)
        out2 = self.relu(out2)

        out3 = self.conv_layer3(out)
        out3 = self.relu(out3)

        out4 = self.conv_layer4(out)
        out4 = self.relu(out4)

        out1 = torch.transpose(out1, 1, 2)
        out2 = torch.transpose(out2, 1, 2)
        out3 = torch.transpose(out3, 1, 2)
        out4 = torch.transpose(out4, 1, 2)

        # global max pool
        out1 = out1.max(dim=1)[0]
        out2 = out2.max(dim=1)[0]
        out3 = out3.max(dim=1)[0]
        out4 = out4.max(dim=1)[0]

        # concantenate all the outputs together
        out     = torch.cat((out1, out2, out3, out4), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment11(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, hidden_dim, num_classes):
        super(Experiment11, self).__init__()

        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.nfms        = [150, 150, 150]
        self.ks          = [4, 4, 5]
        self.Cout        = 2

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # non-static embedding layer ( specific to the current task )
        self.ns_static_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.ns_static_embedding.weight = nn.Parameter(pre_trained_embeddings)

        self.conv_layer1 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[0], self.ks[0])
        self.conv_layer2 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[1], self.ks[1])
        self.conv_layer3 = nn.Conv1d(self.Cout * self.hidden_dim, self.nfms[2], self.ks[2])

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(self.nfms[0] + self.nfms[1] +\
                                     self.nfms[2], self.num_classes)

        self.dropout         = nn.Dropout(0.1)
        self.spatial_dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        static_emb = self.static_embedding(x)
        static_emb = torch.transpose(static_emb, 1, 2)

        ns_static_emb = self.ns_static_embedding(x)
        ns_static_emb = torch.transpose(ns_static_emb, 1, 2)

        out  = torch.cat((static_emb, ns_static_emb), dim = 1)

        # spatial dropout
        out  = self.spatial_dropout(out)

        #out = static_emb

        # pass it through convolutional layer to calculate unigrams
        out1 = self.conv_layer1(out)
        out1 = self.relu(out1)

        out2 = self.conv_layer2(out)
        out2 = self.relu(out2)

        out3 = self.conv_layer3(out)
        out3 = self.relu(out3)

        out1 = torch.transpose(out1, 1, 2)
        out2 = torch.transpose(out2, 1, 2)
        out3 = torch.transpose(out3, 1, 2)

        # global max pool
        out1 = out1.max(dim=1)[0]
        out2 = out2.max(dim=1)[0]
        out3 = out3.max(dim=1)[0]

        # concantenate all the outputs together
        out     = torch.cat((out1, out2, out3), dim=1)

        # apply dropout layer
        out     = self.dropout(out)

        # fully connected layer
        out     = self.fc(out)

        return out


class Experiment13(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, embed_size, hidden_dim, num_classes):
        super(Experiment13, self).__init__()

        self.embed_size  = embed_size
        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes


        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # non-static embedding layer ( specific to the current task )
        self.ns_static_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.ns_static_embedding.weight = nn.Parameter(pre_trained_embeddings)

        self.spatial_dropout = nn.Dropout2d(0.2)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_dim)
        self.fc1  = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.fc2     = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, x):
        static_emb = self.static_embedding(x)
        #static_emb = torch.transpose(static_emb, 1, 2)

       # ns_static_emb = self.ns_static_embedding(x)
       # ns_static_emb = torch.transpose(ns_static_emb, 1, 2)

        #out  = torch.cat((static_emb, ns_static_emb), dim = 1)
        out = static_emb


        # spatial dropout
        out  = self.spatial_dropout(out)

        # transpose input to seq, batch, elements
        out   = torch.transpose(out, 0, 1)

        # pass it through the LSTM cell
        out, (ht, ct) = self.lstm(out)

        # global max pool
        out  = out.max(dim=0)[0]

        # apply dropout layer
        out  = self.fc1(out)
        out  = self.dropout(out)

        # fully connected layer
        out  = self.fc2(out)

        return out


class Experiment14(nn.Module):
    def __init__(self, pre_trained_embeddings, vocab_size, embed_size, hidden_dim, num_classes):
        super(Experiment14, self).__init__()

        self.embed_size  = embed_size
        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.num_classes = num_classes

        self.nfms = [150, 150, 150]
        self.ks   = [3, 3, 5]
        self.Cout = 1

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # convolutional layers
        self.conv_layer1 = nn.Conv1d(self.Cout * self.embed_size, self.nfms[0], self.ks[0], padding=1)
        self.conv_layer2 = nn.Conv1d(self.Cout * self.embed_size, self.nfms[1], self.ks[1], padding=1)
        self.conv_layer3 = nn.Conv1d(self.Cout * self.embed_size, self.nfms[2], self.ks[2], padding=2)

        self.spatial_dropout = nn.Dropout2d(0.2)

        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.nfms[0], self.hidden_dim)
        self.fc   = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, x):
        static_emb = self.static_embedding(x)
        out = static_emb

        # spatial dropout
        out  = self.spatial_dropout(out)

        # transpose input before passing it through conv filters
        out  = torch.transpose(out, 1, 2)

        # convolutional layers
        out1 = self.conv_layer1(out)
        out1 = self.relu(out1)

        out2 = self.conv_layer2(out)
        out2 = self.relu(out2)

        out3 = self.conv_layer3(out)
        out3 = self.relu(out3)

        # transpose it back to batch, widht, channels
        out1 = torch.transpose(out1, 1, 2)
        out2 = torch.transpose(out2, 1, 2)
        out3 = torch.transpose(out3, 1, 2)


        out  = torch.cat((out1, out2, out3), dim=1)

        # transpose input to seq, batch, elements
        out   = torch.transpose(out, 0, 1)

        # pass it through the LSTM cell
        out, (ht, ct) = self.lstm(out)

        # global max pool
        out  = out.max(dim=0)[0]

        # fully connected layer
        out  = self.fc(out)

        return out



def get_exp2_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment2(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=7,
                        PAD_IX=PAD_IX).cuda()
    return model

def get_exp3_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment3(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=7,
                        PAD_IX=PAD_IX).cuda()
    return model


def get_exp4_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment4(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=7,
                        PAD_IX=PAD_IX).cuda()
    return model

def get_exp5_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment5(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=7,
                        PAD_IX=PAD_IX).cuda()
    return model

def get_exp6_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment6(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6,
                        PAD_IX=PAD_IX).cuda()
    return model

def get_exp7_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment7(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6,
                        PAD_IX=PAD_IX).cuda()
    return model

def get_exp8_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment8(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6,
                        PAD_IX=PAD_IX).cuda()
    return model

def get_exp9_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment9(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6).cuda()
    return model

def get_exp10_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment10(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6).cuda()
    return model

def get_exp11_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment11(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6).cuda()
    return model

def get_exp12_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment12(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        hidden_dim=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        num_classes=6).cuda()
    return model

def get_exp13_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment13(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        embed_size=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        hidden_dim=PARAMS[exp_name]['HIDDEN_DIM'],
                        num_classes=6).cuda()
    return model

def get_exp14_model(embedding_matrix, token_to_id, exp_name, PAD_IX):
    model = Experiment14(pre_trained_embeddings=torch.FloatTensor(embedding_matrix),
                        vocab_size=len(token_to_id),
                        embed_size=PARAMS[exp_name]['EMBEDDING_SIZE'],
                        hidden_dim=PARAMS[exp_name]['HIDDEN_DIM'],
                        num_classes=6).cuda()
    return model
