import torch
import numpy as np
import logging
import os
import torch.nn.functional as F
import data_reader.kaldi_io as ko
import data_reader.adv_kaldi_io as ako
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def train(args, model, device, train_loader, optimizer, epoch, rnn=False):
    model.train()
    for batch_idx, (_, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.view(-1,1).float()
        optimizer.zero_grad()
        if rnn == True:
            model.hidden = model.init_hidden(data.size()[0]) # clear out the hidden state of the LSTM
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
"""
def train(args, model, device, train_loader, optimizer, epoch, train_scp, train_utt2label, plot_wd, rnn=False):
    model.train()
    for batch_idx, (id_list, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.view(-1,1).float()
        optimizer.zero_grad()
        if rnn == True:
            model.hidden = model.init_hidden(data.size()[0]) # clear out the hidden state of the LSTM
        output, weight = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        mat = weight # heatmap of the weight 
        mat = mat.data.cpu().numpy()
        for i in range(len(id_list)):
            make_plot(id_list[i], mat[i], plot_wd, epoch) 

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def make_plot(utt_id, mat, plot_wd, epoch):
    mat = mat.reshape(257,1091)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,4))
    cax = ax.matshow(mat, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    fig.colorbar(cax)
    plt.savefig(plot_wd + utt_id + '-' + str(epoch) + '.png')
"""

def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    if is_best:
        torch.save(state, snapshot_file)
        logger.info("Snapshot saved to {}\n".format(snapshot_file))
