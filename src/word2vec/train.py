import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
import time

from utils import *
from model import *
from visualization import *
from losses import *

def train_w2v(contexts, V, num_skips, batch_size):
    loss_every_nsteps = config.LOGGING_STEPS
    total_loss = 0
    start_time = time.time()
    model      = create_model(V)

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer     = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    for step, (batch, labels) in enumerate(make_skip_gram_batchs_iter(contexts, config.WINDOW_SIZE, num_skips=num_skips, batch_size=batch_size)):
        batch  = torch.cuda.LongTensor(batch)
        labels = torch.cuda.LongTensor(labels)

        logits = model(batch)
        loss   = loss_function(logits, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.cpu().detach().item()

        if step != 0 and step % loss_every_nsteps == 0:
            print("Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(step, total_loss / loss_every_nsteps,\
                                                                                    time.time() - start_time))
            total_loss = 0
            start_time = time.time()

    return model


def single_matrix_loss(embeddings, input_word_repr, labels):
    logits  = torch.mm(input_word_repr, torch.transpose(embeddings, 1, 0))
    loss_fn = nn.CrossEntropyLoss().cuda()
    loss    = loss_fn(logits, labels)

    return loss

def train_w2v_one_matrix(contexts, V, num_skips, batch_size):
    loss_every_nsteps = config.LOGGING_STEPS
    total_loss = 0
    start_time = time.time()
    model      = create_model_v2(V)
    
    
    optimizer  = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    for step, (batch, labels) in enumerate(make_skip_gram_batchs_iter(contexts, config.WINDOW_SIZE, num_skips=num_skips, batch_size=batch_size)):
        batch  = torch.cuda.LongTensor(batch)
        labels = torch.cuda.LongTensor(labels)

        input_word_repr = model(batch)
        embeddings      = model[0].weight
        loss            = single_matrix_loss(embeddings, input_word_repr, labels)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.cpu().detach().item()

        if step != 0 and step % loss_every_nsteps == 0:
            print("Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(step, total_loss / loss_every_nsteps,\
                                                                                    time.time() - start_time))
            total_loss = 0
            start_time = time.time()

    return model

def train_w2v_vanilla_neg_sampling(contexts, V, num_skips, batch_size, word2index, p_w):
    loss_every_nsteps = config.LOGGING_STEPS
    total_loss = 0
    start_time = time.time()
    model      = create_model(V)

    optimizer  = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    for step, (batch, labels) in enumerate(make_skip_gram_batchs_iter(contexts, config.WINDOW_SIZE, num_skips=num_skips, batch_size=batch_size)):
        batch  = torch.cuda.LongTensor(batch)
        labels = torch.cuda.LongTensor(labels)

        logits = model(batch)
        loss   = vanilla_neg_sampling_loss(logits, labels, word2index, p_w)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.cpu().detach().item()

        if step != 0 and step % loss_every_nsteps == 0:
            print("Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(step, total_loss / loss_every_nsteps,\
                                                                                    time.time() - start_time))
            total_loss = 0
            start_time = time.time()

    return model

def train_w2v_batch_transpose_trick(contexts, V, num_skips, batch_size):
    loss_every_nsteps = config.LOGGING_STEPS
    total_loss = 0
    start_time = time.time()
    model      = create_model_v2(V)

    optimizer  = optim.Adam(model.parameters(), lr=config.BTT_LEARNING_RATE)


    for step, (batch, labels) in enumerate(make_skip_gram_batchs_iter_with_context(contexts, config.WINDOW_SIZE, num_skips=num_skips, batch_size=batch_size)):
        batch  = torch.cuda.LongTensor(batch)
        labels = torch.cuda.LongTensor(labels)

        center_word_repr = model(batch)

        # normalized central word representation
        center_word_repr_norm = center_word_repr / torch.norm(center_word_repr, p=2)
        loss                  = batch_transpose_trick_loss(center_word_repr_norm, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.cpu().detach().item()

        if step != 0 and step % loss_every_nsteps == 0:
            print("Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(step, total_loss / loss_every_nsteps,\
                                                                                    time.time() - start_time))
            total_loss = 0
            start_time = time.time()

    return model


