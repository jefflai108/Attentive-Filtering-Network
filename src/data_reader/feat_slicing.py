from __future__ import print_function
import os
import numpy as np
import kaldi_io as ko

"""
Reads a kaldi scp file and slice the feature matrix

Jeff, 2018
"""

def tensor_cnn_frame(mat, M):
    """Construct a tensor of shape (C x H x W) given an utterance matrix 
    for CNN
    """
    slice_mat = []
    for index in np.arange(len(mat)):
        if index < M:
            to_left = np.tile(mat[index], M).reshape((M,-1))
            rest = mat[index:index+M+1]
            context = np.vstack((to_left, rest))
        elif index >= len(mat)-M:
            to_right = np.tile(mat[index], M).reshape((M,-1))
            rest = mat[index-M:index+1]
            context = np.vstack((rest, to_right))
        else:
            context = mat[index-M:index+M+1]
        slice_mat.append(context)

    slice_mat = np.array(slice_mat)
    slice_mat = np.expand_dims(slice_mat, axis=1)
    slice_mat = np.swapaxes(slice_mat, 2, 3)
    
    return slice_mat

def tensor_cnn_utt(mat, max_len):
    mat = np.swapaxes(mat, 0, 1)
    repetition = int(max_len/mat.shape[1])
    tensor = np.tile(mat,repetition)
    repetition = max_len % mat.shape[1]
    rest = mat[:,:repetition]
    tensor = np.hstack((tensor,rest))
    #tensor = np.expand_dims(tensor, axis=0)
    return tensor

def tensor_cnngru(mat):
    """Construct an utterance tensor for a given utterance matrix mat
    for CNN+GRU
    """
    mat = np.swapaxes(mat, 0, 1)
    div = int(mat.shape[1]/400)
    if div == 0: # short utt
        tensor_mat = mat
        while True:
            shape = tensor_mat.shape[1]
            if shape + mat.shape[1] < 400:
                tensor_mat = np.hstack((tensor_mat,mat))
            else:
                tensor_mat = np.hstack((tensor_mat,mat[:,:400-shape]))
                break
    elif div == 1: # truncate to 1
        tensor_mat = mat[:,:400]
    else:
        # TO DO: cut into 2
        tensor_mat = mat[:,:400]

    tensor_mat = np.expand_dims(tensor_mat, axis=2)
    print(tensor_mat.shape)
    return tensor_mat

def slice(mat, M):
    """Slice a feature matrix with context M
    for feed-forward DNN
    """
    slice_mat = []
    for index in np.arange(len(mat)):
        if index < M:
            to_left = np.tile(mat[index], M).reshape((M,-1))
            rest = mat[index:index+M+1]
            context = np.vstack((to_left, rest))
        elif index >= len(mat)-M:
            to_right = np.tile(mat[index], M).reshape((M,-1))
            rest = mat[index-M:index+1]
            context = np.vstack((rest, to_right))
        else:
            context = mat[index-M:index+M+1]
        slice_mat.append(context)

    slice_mat = np.array(slice_mat)
    slice_mat = slice_mat.reshape((slice_mat.shape[0],-1))

    return slice_mat

def write_kaldi(orig_feat_scp, ark_scp_output, max_len):
    """Write the slice feature matrix to ark_scp_output
    """
    with ko.open_or_fd(ark_scp_output,'wb') as f:
        for key,mat in ko.read_mat_scp(orig_feat_scp):
            tensor = tensor_cnn_utt(mat, max_len)
            if tensor.shape[1] != max_len:
                print(tensor.shape)
            ko.write_mat(f, tensor, key=key)

def calculate_len(train_scp, dev_scp, eval_scp):
    shortest = np.inf
    longest  = -np.inf
    total    = 0
    counter  = 0 

    for i in [train_scp, dev_scp, eval_scp]:
        for key,mat in ko.read_mat_scp(i):
            curr = len(mat)
            shortest = min(curr,shortest)
            longest  = max(curr,longest)
            total += curr 
            counter += 1
        print(i, total*1./counter)

    print(shortest, longest, total*1./counter)

if __name__ == '__main__':

    data_wd = 'cqcc/'
    curr_wd = os.getcwd()
    orig_train_scp  = curr_wd + '/' + data_wd + 'train_cqcc_spectrogram_cmvn_orig.scp'
    orig_dev_scp    = curr_wd + '/' + data_wd + 'dev_cqcc_spectrogram_cmvn_orig.scp'
    orig_eval_scp   = curr_wd + '/' + data_wd + 'eval_cqcc_spectrogram_cmvn_orig.scp'

    write_wd = 'cqcc/'
    out_train_scp  = curr_wd + '/' + write_wd + 'train_cqcc_spectrogram_cmvn_tensor.scp'
    out_train_ark  = curr_wd + '/' + write_wd +'train_cqcc_spectrogram_cmvn_tensor.ark'
    out_dev_scp    = curr_wd + '/' + write_wd + 'dev_cqcc_spectrogram_cmvn_tensor.scp'
    out_dev_ark    = curr_wd + '/' + write_wd +'dev_cqcc_spectrogram_cmvn_tensor.ark'
    out_eval_scp   = curr_wd + '/' + write_wd + 'eval_cqcc_spectrogram_cmvn_tensor.scp'
    out_eval_ark   = curr_wd + '/' + write_wd +'eval_cqcc_spectrogram_cmvn_tensor.ark'

    train_ark_scp='ark:| copy-feats --compress=true ark:- ark,scp:' + out_train_ark + ',' + out_train_scp
    dev_ark_scp='ark:| copy-feats --compress=true ark:- ark,scp:' + out_dev_ark + ',' + out_dev_scp
    eval_ark_scp='ark:| copy-feats --compress=true ark:- ark,scp:' + out_eval_ark + ',' + out_eval_scp

    ## calculate statistics 
    #calculate_len(orig_train_scp, orig_dev_scp, orig_eval_scp)
    
    ## logspec without vad:
    #max_len = 1091
    #min_len = 58
    #avg_len = 282
    ## logspec with vad:
    #max_len = 1091
    #min_len = 69
    #avg_len = 312
    ## cqcc_spectrogram without vad:
    max_len = 1278
    min_len = 80
    avg_len = 365

    truncate_len = max_len
    write_kaldi(orig_train_scp, train_ark_scp, truncate_len)
    write_kaldi(orig_dev_scp, dev_ark_scp, truncate_len)
    write_kaldi(orig_eval_scp, eval_ark_scp, truncate_len)


