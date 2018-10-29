from __future__ import print_function
import os 
import torch
import numpy as np
import logging
import torch.nn.functional as F
from v1_metrics import compute_eer
import data_reader.kaldi_io as ko
import data_reader.adv_kaldi_io as ako
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def retrieve_weight(args, model, device, eval_loader, eval_scp, eval_utt2label, plot_wd, rnn=False):
    """get the attention weight, batch size has to be 1
    """
    logger.info("Starting evaluation")
    logger.info("plot saves at {}".format(plot_wd))

    model.eval()
    with torch.no_grad():
        for id_list, data, target in eval_loader:
            assert len(id_list) == 1, 'batch size has to set to 1'
            data, target = data.to(device), target.to(device)
            target = target.view(-1,1).float()
            #_, weight = model(data) 
            #mat = weight # heatmap of the weight 
            #mat = data + data * weight 
            mat = data
            mat = mat.data.cpu().numpy()
            make_plot(id_list[0], mat, plot_wd) 

def make_plot(utt_id, mat, plot_wd):
    """plot logspec with attention 
    """
    mat = mat.reshape(257,1091)
    #mat = mat.T
    #mat = mat.reshape(257,160)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,4))
    cax = ax.matshow(mat, interpolation='nearest', aspect='auto', cmap=plt.cm.YlOrBr, vmin=0.0, vmax=1.0, origin='lower')
    fig.colorbar(cax)
    plt.savefig(plot_wd + utt_id + '.pdf', bbox_inches='tight')


