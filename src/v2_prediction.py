import torch
import numpy as np
import logging
import torch.nn.functional as F
from v1_metrics import compute_eer, compute_confuse
import data_reader.adv_kaldi_io as ako
from v2_validation import utt_eer, compute_loss

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def prediction(args, model, device, eval_loader, eval_scp, eval_utt2label, opti_threshold):
    logger.info("Starting evaluation")
    utt2len   = ako.read_key_len(eval_scp)
    utt2label = ako.read_key_label(eval_utt2label)
    key_list  = ako.read_all_key(eval_scp)
    eval_loss, eval_scores, eval_correct = compute_loss(model, device, eval_loader)
    eval_eer = utt_eer(eval_scores, utt2len, utt2label, key_list, opti_threshold)
    
    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> evalidation set: Average loss: {:.4f}\tEER: {:.4f}\n'.format(
                eval_loss, eval_eer))


    return eval_loss, eval_eer

