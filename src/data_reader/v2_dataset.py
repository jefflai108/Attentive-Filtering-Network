import numpy as np
import torch
from torch.utils import data
import adv_kaldi_io as ako
import kaldi_io as ko

"""
Preloads all the features (without context)

"""

class SpoofDataset(data.Dataset):
    """PyTorch dataset that reads kaldi feature
    """
    def __init__(self, scp_file, utt2label_file):
        'Initialization'
        self.scp_file  = scp_file
        self.utt2label = ako.read_key_label(utt2label_file)
        self.feat_gen  = ko.read_mat_scp(scp_file) # feature generator

        mats, labels = [], [] # construct feature and label matrices
        for key,mat in self.feat_gen:
            mats.append(mat)
            labels.append(np.repeat(self.utt2label[key], len(mat)))
        self.label_mat  = np.hstack(labels)
        self.feat_mat   = np.vstack(mats)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_mat)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.feat_mat[index]
        y = self.label_mat[index]

        return X, y
