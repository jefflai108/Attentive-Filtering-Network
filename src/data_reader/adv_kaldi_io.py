from __future__ import print_function
import kaldi_io as ko
"""Supplement on top of kaldi_io for python,
but first concatenate all feature scp files into one big fat scp file before using this script

Jeff Lai, 2018
"""

def read_all_key(file):
    """return all keys/utterances of a kaldi scp file
    """
    key_list = []
    fd = ko.open_or_fd(file)
    try:
        for line in fd:
            (key,_) = line.decode().split(' ')
            key_list.append(key)
    finally:
        if fd is not file: fd.close()
        return key_list

def read_mat_key(file, target_key):
    """read the matrix of the target key/utterance from a kaldi scp file
    """
    fd = ko.open_or_fd(file)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            if key == target_key:
                return ko.read_mat(rxfile)
    finally:
        if fd is not file: fd.close()

def read_total_len(file):
    """return length of all keys in a kaldi scp file
    """
    total_len = 0
    key_map = read_key_len(file)
    for value in key_map.values():
        total_len += value
    return total_len

def read_key_len(file):
    """return a dictionary of key --> length of the key
    for all keys in a kaldi scp file
    """
    key_map = {}
    key_list = read_all_key(file)
    for key in key_list:
        mat = read_mat_key(file, key)
        key_map[key] = len(mat)
    return key_map

def read_key_label(file):
    """return a dictionary of key --> label of the key, where
    genuine: 1
    spoof: 0

    for all keys in a kaldi scp file
    """
    utt2label = {}
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for i in content:
        key = i.split()[0]
        label = i.split()[1]
        if label == 'genuine':
            utt2label[key] = 1
        else:
            utt2label[key] = 0

    return utt2label


