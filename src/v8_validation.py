import numpy as np
import logging
from timeit import default_timer as timer
from scipy.optimize import fmin_l_bfgs_b, basinhopping
import torch
import torch.nn.functional as F
from v1_metrics import compute_eer
import data_reader.adv_kaldi_io as ako

"""
utterance-based validation without stochastic search for threshold 
important: EER does not need a threshold. 
"""

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def validation(args, model, device, val_loader, val_scp, val_utt2label):
    logger.info("Starting Validation")
    val_loss, val_scores = compute_loss(model, device, val_loader)
    val_preds, val_labels =  utt_scores(val_scores, val_scp, val_utt2label)
    val_eer  = compute_eer(val_labels, val_preds)

    logger.info('===> Validation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                val_loss, val_eer))
    return val_loss, val_eer

def utt_scores(scores, scp, utt2label):
    """return predictions and labels per utterance
    """
    utt2label = ako.read_key_label(utt2label)

    preds, labels = [], []
    for key,value in scores.iteritems():
        preds.append(value)
        labels.append(utt2label[key])

    return np.array(preds), np.array(labels)

def compute_loss(model, device, data_loader):
    model.eval()
    loss = 0
    scores  = {}

    with torch.no_grad():
        for id_list, X1, X2, target in data_loader:
            X1, X2, target = X1.to(device), X2.to(device), target.to(device)
            target = target.view(-1,1).float()
            y = model(X1, X2)
            loss += F.binary_cross_entropy(y, target, size_average=False)
            for i,id in enumerate(id_list):
                scores[id] = y[i].data.cpu().numpy()

    loss /= len(data_loader.dataset) # average loss

    return loss, scores

