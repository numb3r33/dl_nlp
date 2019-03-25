import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from models import *
from utils import *
from config import *

SEED = 41
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

def do_epoch(model, criterion, data, batch_size, optimizer=None):
    epoch_loss, total_size = 0, 0
    per_label_preds = [[], [], [], [], [], []]
    per_label_true  = [[], [], [], [], [], []]

    is_train = not optimizer is None
    model.train(is_train)

    data, labels = data
    batchs_count = math.ceil(data.shape[0] / batch_size)

    with torch.autograd.set_grad_enabled(is_train):
        for i, (X_batch, y_batch) in enumerate(iterate_batches(data, labels, batch_size)):
            X_batch, y_batch = torch.cuda.LongTensor(X_batch), torch.cuda.FloatTensor(y_batch)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # convert true target
            batch_target = y_batch.cpu().detach().numpy()
            logits_cpu   = logits.cpu().detach().numpy()

            # per_label_preds
            for j in range(6):
                label_preds     = logits_cpu[:, j]
                per_label_preds[j].extend(label_preds)
                per_label_true[j].extend(batch_target[:, j])

            # calculate log loss
            epoch_loss += loss.item()

            print('\r[{} / {}]: Loss = {:.4f}'.format(
                  i, batchs_count, loss.item(), end=''))

    label_auc = []

    for i in range(6):
        label_auc.append(roc_auc_score(per_label_true[i], per_label_preds[i]))

    return epoch_loss / batchs_count, np.mean(label_auc)

def fit(model, criterion, optimizer, train_data, epochs_count=1,
        batch_size=32, val_data=None, val_batch_size=None):
    if not val_data is None and val_batch_size is None:
        val_batch_size = batch_size

    for epoch in range(epochs_count):
        start_time = time.time()
        train_loss, train_auc = do_epoch(
            model, criterion, train_data, batch_size, optimizer
        )

        output_info = '\rEpoch {} / {}, Epoch Time = {:.2f}s: Train Loss = {:.4f}, Train AUC = {:.4f}'

        if not val_data is None:
            val_loss, val_auc   = do_epoch(model, criterion, val_data, val_batch_size, None)

            epoch_time   = time.time() - start_time
            output_info += ', Val Loss = {:.4f}, Val AUC = {:.4f}'
            print(output_info.format(epoch+1, epochs_count, epoch_time,
                                     train_loss,
                                     train_auc,
                                     val_loss,
                                     val_auc
                                    ))
        else:
            epoch_time = time.time() - start_time
            print(output_info.format(epoch+1, epochs_count, epoch_time, train_loss, train_auc))

    return model

def train_and_evaluate(model, criterion, optimizer, embedding_matrix, token_to_id, exp_name, data_train, data_val, TARGET_COLS, UNK_IX, PAD_IX, run_mode):
    if run_mode == 'train':
        X_train      = as_matrix(data_train['tokenized_comments'], token_to_id, PARAMS[exp_name]['WORD_DROPOUT'], UNK_IX, PAD_IX, max_len=PARAMS[exp_name]['MAX_LEN'])
        train_labels = data_train.loc[:, TARGET_COLS].values

        X_test       = as_matrix(data_val['tokenized_comments'], token_to_id, PARAMS[exp_name]['WORD_DROPOUT'], UNK_IX, PAD_IX, max_len=PARAMS[exp_name]['MAX_LEN'])
        test_labels  = data_val.loc[:, TARGET_COLS].values
    else:
        X_train      = as_matrix(data_train['tokenized_comments'], token_to_id, PARAMS[exp_name]['WORD_DROPOUT'], UNK_IX, PAD_IX, max_len=PARAMS[exp_name]['MAX_LEN'])
        train_labels = data_train.loc[:, TARGET_COLS].values

        X_test       = as_matrix(data_val['tokenized_comments'], token_to_id, PARAMS[exp_name]['WORD_DROPOUT'], UNK_IX, PAD_IX, max_len=PARAMS[exp_name]['MAX_LEN'])

    if run_mode == 'train':
        model        = fit(model, criterion, optimizer, train_data=(X_train, train_labels), epochs_count=PARAMS[exp_name]['NB_EPOCHS'],
                       batch_size=PARAMS[exp_name]['BATCH_SIZE'], val_data=(X_test, test_labels), val_batch_size=1024)
        return model, []
    else:
        model        = fit(model, criterion, optimizer, train_data=(X_train, train_labels), epochs_count=PARAMS[exp_name]['NB_EPOCHS'],
                       batch_size=PARAMS[exp_name]['BATCH_SIZE'], val_batch_size=1024)
        preds        = predict(model, X_test, batch_size=1024)

        return model, preds

def predict(model, data, batch_size):
    is_train = False
    model.train(is_train)

    batchs_count = math.ceil(data.shape[0] / batch_size)
    preds        = []

    with torch.autograd.set_grad_enabled(is_train):
        for i, X_batch in enumerate(iterate_batches(data, labels=[], batch_size=batch_size, predict_mode='test')):
            X_batch = torch.cuda.LongTensor(X_batch)
            logits  = model(X_batch)
            p       = torch.sigmoid(logits).cpu().detach().numpy()

            preds.append(p)

    return np.vstack(preds)

def prepare_submission(test_labels, preds, fp):
    df = test_labels.copy()
    df.iloc[:, 1:] = preds
    df.to_csv(fp, index=False)

    return df
