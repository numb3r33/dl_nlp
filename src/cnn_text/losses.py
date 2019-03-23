import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import *

def get_exp2_criterion():
    return nn.BCEWithLogitsLoss().cuda()

def get_exp2_optimizer(model, exp_name):
    return  optim.Adam([param for param in model.parameters() if param.requires_grad], lr=PARAMS[exp_name]['LEARNING_RATE'])

