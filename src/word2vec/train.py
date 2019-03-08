import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
import time

from utils import *
from model import *


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
