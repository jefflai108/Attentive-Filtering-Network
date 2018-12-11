import torch
import numpy as np
from torch.utils import data
import adv_kaldi_io as ako
import kaldi_io as ko

"""
Preloads the original feature and constructs a tensor of shape (C x H x W) on the fly
for CNN

Pros: requires less memory 
Cons: too slow...
"""

class SpoofDataset(data.Dataset):
    """PyTorch dataset that reads kaldi feature
    """
    def __init__(self, scp_file, utt2label_file, M):
        'Initialization'
        self.M = M
        self.scp_file  = scp_file
        self.utt2len   = ako.read_key_len(scp_file)
        self.utt2label = ako.read_key_label(utt2label_file)
        self.feat_gen  = ko.read_mat_scp(scp_file) # feature generator

        self.utt2mat = {}
        for key,mat in self.feat_gen:
            self.utt2mat[key] = mat

    def __len__(self):
        'Denotes the total number of samples'
        return ako.read_total_len(self.scp_file)

    def __getitem__(self, index):
        'Generates one sample of data'
        # get the utterance the index belongs to
        curr_idx = 0
        for key, value in self.utt2len.iteritems():
            curr_idx += value
            if index < curr_idx:
                utt_id = key
                index = index-(curr_idx-value)
                break
        target_mat = self.utt2mat[utt_id] # get the matrix

        # construct context on the fly
        if index < self.M:
            to_left = np.tile(target_mat[index], self.M).reshape((self.M,-1))
            rest = target_mat[index:index+self.M+1]
            context = np.vstack((to_left, rest))
        elif index >= len(target_mat)-self.M:
            to_right = np.tile(target_mat[index], self.M).reshape((self.M,-1))
            rest = target_mat[index-self.M:index+1]
            context = np.vstack((rest, to_right))
        else:
            context = target_mat[index-self.M:index+self.M+1]
        context = np.expand_dims(context, axis=0)
        X = np.swapaxes(context, 1, 2)

        # Load data and get label
        y = self.utt2label[utt_id]
        
        return X, y
