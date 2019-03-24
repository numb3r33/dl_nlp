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

        self.Cin         = 32
        self.Cout        = 1

        # first embedding layer that is static ( non-trainable )
        self.static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        self.static_embedding.weight = nn.Parameter(pre_trained_embeddings)
        # make embedding layer non-trainable
        self.static_embedding.weight.requires_grad = False

        # second embedding layer that is not-static ( both of them are pre-trained embeddings )
        #self.non_static_embedding        = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_IX)
        #self.non_static_embedding.weight = nn.Parameter(pre_trained_embeddings)


        self.conv_layer1 = nn.Conv2d(self.Cout, self.Cin, kernel_size=(3, self.hidden_dim))

        self.relu        = nn.ReLU()
        self.fc          = nn.Linear(1 * self.Cin, self.num_classes)

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
