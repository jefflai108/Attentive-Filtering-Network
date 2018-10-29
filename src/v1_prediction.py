import torch
import numpy as np
import logging
import torch.nn.functional as F
from v1_metrics import compute_eer, compute_confuse
import data_reader.adv_kaldi_io as ako
from v1_validation import compute_loss

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def prediction(args, model, device, eval_loader, eval_scp, eval_utt2label, opti_threshold):
    logger.info("Starting evaluation")
    eval_loss, eval_scores, eval_correct = compute_loss(model, device, eval_loader)
    eval_eer, eval_confuse_mat = compute_utt_eer(eval_scores, eval_scp, eval_utt2label, opti_threshold)

    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> evalidation set: Average loss: {:.4f}\tEER: {:.4f}\tConfusion Matrix: {}\n'.format(
                eval_loss, eval_eer, eval_confuse_mat))

    return eval_loss, eval_eer

def compute_utt_eer(scores, scp, utt2label, threshold):
    """utterance-based eer
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
        if avg_scores < threshold:
            preds.append(0)
        else: preds.append(1)
        labels.append(utt2label[key])

    eer = compute_eer(labels, preds)
    confuse_mat = compute_confuse(labels, preds)
    return eer, confuse_mat

