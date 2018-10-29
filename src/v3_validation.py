import numpy as np
import logging
from timeit import default_timer as timer
from scipy.optimize import fmin_l_bfgs_b, basinhopping
import torch
import torch.nn.functional as F
from v1_metrics import compute_eer
import data_reader.adv_kaldi_io as ako

"""
validation without stochastic search for threshold 
important: EER does not need a threshold. 
"""

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def validation(args, model, device, train_loader, train_scp, train_utt2label, val_loader, val_scp, val_utt2label):
    logger.info("Starting Validation")
    train_loss, train_scores = compute_loss(model, device, train_loader)
    val_loss, val_scores = compute_loss(model, device, val_loader)
    train_preds, train_labels = utt_scores(train_scores, train_scp, train_utt2label) 
    val_preds, val_labels =  utt_scores(val_scores, val_scp, val_utt2label)
    train_eer = compute_eer(train_labels, train_preds)
    val_eer  = compute_eer(val_labels, val_preds)

    logger.info('===> Training set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                train_loss, train_eer))
    logger.info('===> Validation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                val_loss, val_eer))
    return val_loss, val_eer

def utt_scores(scores, scp, utt2label):
    """return predictions and labels per utterance
    """
    utt2len   = ako.read_key_len(scp)
    utt2label = ako.read_key_label(utt2label)
    key_list  = ako.read_all_key(scp)

    preds, labels = [], []
    idx = 0
    for key in key_list:
        frames_per_utt = utt2len[key]
        avg_scores = np.average(scores[idx:idx+frames_per_utt])
        idx = idx + frames_per_utt
        preds.append(avg_scores)
        labels.append(utt2label[key])

    return np.array(preds), np.array(labels)

def compute_loss(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    scores  = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1,1).float()
            #output, hidden = model(data, None)
            output = model(data)
            loss += F.binary_cross_entropy(output, target, size_average=False)

            scores.append(output.data.cpu().numpy())

    loss /= len(data_loader.dataset) # average loss
    scores = np.vstack(scores) # scores per frame

    return loss, scores

