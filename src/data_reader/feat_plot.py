from __future__ import print_function
import os
import numpy as np
import kaldi_io as ko
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

"""
Reads a kaldi scp file and plot the spectrogram 

Jeff, 2018
"""

def plot_feat(orig_feat_scp, output_plot_wd):
    """plot one Kaldi logspec feat 
    """
    for key,mat in ko.read_mat_scp(orig_feat_scp):
        #mat = np.transpose(mat)
        print(mat.shape)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,4))
        cax = ax.matshow(mat, interpolation='nearest', aspect='auto', 
                cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.savefig(output_plot_wd + key + '.png')

if __name__ == '__main__':
    data_wd = 'spec/'
    curr_wd = os.getcwd()
    orig_train_scp  = curr_wd + '/' + data_wd + 'train_spec_cmvn_tensor.scp'
    orig_dev_scp    = curr_wd + '/' + data_wd + 'dev_spec_cmvn_tensor.scp'
    orig_eval_scp   = curr_wd + '/' + data_wd + 'eval_spec_cmvn_orig.scp'

    #orig_train_scp  = curr_wd + '/' + data_wd + 'train_spec_vad_cmvn_tensor.scp'
    #orig_train_ark  = curr_wd + '/' + data_wd +'train_spec_vad_cmvn_tensor.ark'
    #orig_dev_scp    = curr_wd + '/' + data_wd + 'dev_spec_vad_cmvn_tensor.scp'
    #orig_dev_ark    = curr_wd + '/' + data_wd +'dev_spec_vad_cmvn_tensor.ark'
    #orig_eval_scp   = curr_wd + '/' + data_wd + 'eval_spec_vad_cmvn_tensor.scp'
    #orig_eval_ark   = curr_wd + '/' + data_wd +'eval_spec_vad_cmvn_tensor.ark'

    plot_wd = 'plot/spec/train_tensor/'
    plot_feat(orig_train_scp, plot_wd)

