import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import time

from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from utils import check_labels, readable_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion):
    
    epoch_loss      = 0
    per_label_preds = [[], [], [], [], [], []]
    per_label_true  = [[], [], [], [], [], []]
 
    model.train()
    
    for i, batch in enumerate(iterator):
        
        optimizer.zero_grad()
        X, y        = batch        
        
        X           = X.to(device)
        y           = y.to(device)
        
        predictions = model(X)
        
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        # convert true target
        batch_target = y.cpu().detach().numpy()
        logits_cpu   = predictions.cpu().detach().numpy()

        # per_label_preds
        for j in range(6):
            label_preds     = logits_cpu[:, j]
            per_label_preds[j].extend(label_preds)
            per_label_true[j].extend(batch_target[:, j])

        # calculate log loss
        epoch_loss += loss.item()

        print('\r[{} / {}]: Loss = {:.4f}'.format(
              i, len(iterator), loss.item(), end=''))
    
    label_auc = []

    for i in range(6):
        label_auc.append(roc_auc_score(per_label_true[i], per_label_preds[i]))
    
    return epoch_loss / len(iterator), np.mean(label_auc)


def evaluate(model, iterator, criterion):
    epoch_loss      = 0
    per_label_preds = [[], [], [], [], [], []]
    per_label_true  = [[], [], [], [], [], []]
    preds           = []

    model.eval()
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            X, y        = batch
            
            X           = X.to(device)
            predictions = model(X)
            
            # convert true target
            logits_cpu   = predictions
            
            if not check_labels(y): 
                y    = y.to(device)
                loss = criterion(predictions, y)
                batch_target = y.cpu().detach().numpy()      
                logits_cpu = logits_cpu.cpu().detach().numpy()

                preds.append(logits_cpu)

                # per_label_preds
                for j in range(6):
                    label_preds     = logits_cpu[:, j]
                    per_label_preds[j].extend(label_preds)
                    per_label_true[j].extend(batch_target[:, j])

                # calculate log loss
                epoch_loss += loss.item()

                print('\r[{} / {}]: Loss = {:.4f}'.format(
                      i, len(iterator), loss.item(), end=''))
            else:
                probs = torch.sigmoid(logits_cpu).cpu().detach().numpy()
                preds.append(probs)
    
    label_auc = []

    if len(per_label_preds[0]) > 0:
        for i in range(6):
            label_auc.append(roc_auc_score(per_label_true[i], per_label_preds[i]))

    return epoch_loss / len(iterator), np.mean(label_auc) if len(label_auc) > 0 else 0, np.vstack(preds)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def learn(model, trn_dl, vld_dl, vocab, config):
    vocab_size = len(vocab.itos)
    lr         = config['lr']

    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.BCEWithLogitsLoss()

    model      = model.to(device)
    criterion  = criterion.to(device)

    N_EPOCHS        = config['N_EPOCHS']
    best_valid_loss = float('inf')
    time_identifier = readable_time()

    for epoch in range(N_EPOCHS):
        start_time = time.time()
    
        train_loss, train_auc    = train(model, trn_dl, optimizer, criterion)
        if vld_dl is not None: valid_loss, valid_auc, _ = evaluate(model, vld_dl, criterion)
    
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if vld_dl is not None:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model      = model
                
                print('Saving best model found so far to disk')
                # save model to results directory
                torch.save(model.state_dict(), config['result_dir'] + config['model_name'] + time_identifier  +'.pth')
                joblib.dump(config, config['result_dir'] + config['model_name'] + 'config_' + time_identifier + '.pkl')
    
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train AUC: {train_auc:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. AUC: {valid_auc:.3f}') 
        else:
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train AUC: {train_auc:.3f}')
    
    # save full trained model to disk
    if vld_dl is None:
        torch.save(model.dict(), config['result_dir'] + config['model_name'] + time_identifier +'_full.pth')
        joblib.dump(config, config['result_dir'] + config['model_name'] + 'config_full' + time_identifier + '.pkl')
    
    return model


def predictions(model, tst_dl, criterion, test_labels, fn):
    _, _, final_preds = evaluate(model, tst_dl, criterion=None)

    df = test_labels.copy()
    df.iloc[:, 1:] = final_preds
    df.to_csv(fn, index=False)

    return df
