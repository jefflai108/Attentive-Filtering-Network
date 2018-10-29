import torch
import numpy as np
import logging
import torch.nn.functional as F
from v1_metrics import compute_eer
import data_reader.adv_kaldi_io as ako
from v4_validation import compute_loss, utt_scores

"""
correspond to v3_validation, threshold-less EER 
"""

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def prediction(args, model, device, eval_loader, eval_scp, eval_utt2label, rnn=False):
    logger.info("Starting evaluation")
    eval_loss, eval_scores = compute_loss(model, device, eval_loader, rnn)
    eval_preds, eval_labels = utt_scores(eval_scores, eval_scp, eval_utt2label)
    eval_eer  = compute_eer(eval_labels, eval_preds)

    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> evalidation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                eval_loss, eval_eer))

    return eval_loss, eval_eer

def scores(args, model, device, eval_loader, eval_scp, eval_utt2label, rnn=False):
    """get the scores only (for averaging across models later on)
    """
    #logger.info("Starting evaluation")
    eval_loss, eval_scores = compute_loss(model, device, eval_loader, rnn)
    eval_preds, eval_labels = utt_scores(eval_scores, eval_scp, eval_utt2label)

    return eval_preds, eval_labels

