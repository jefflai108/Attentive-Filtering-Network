import torch
import numpy as np
from torch.utils import data
from v3_dataset import SpoofDataset
import kaldi_io as ko
import adv_kaldi_io as ako

train_utt2label = '/export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/train_utt2label'
validation_utt2label = '/export/b19/jlai/cstr/spoof/model/src/data_reader/data/utt2label/dev_utt2label'
evaluation_utt2label = '/export/b19/jlai/cstr/spoof/model/src/data_reader/spec/utt2label/eval_utt2label'
train_scp = '/export/b19/jlai/cstr/spoof/model/src/data_reader/cqcc/train_cqcc_spectrogram_cmvn_tensor.scp'
validation_scp   = '/export/b19/jlai/cstr/spoof/model/src/data_reader/spec/dev_spec_cmvn_tensor.scp'
evaluation_scp = '/export/b19/jlai/cstr/spoof/model/src/data_reader/spec/eval_spec_cmvn_tensor.scp'

params = {'batch_size': 32,
          'shuffle': False,
          'num_workers': 0}
max_epochs = 1

training_set, validation_set, evaluation_set = SpoofDataset(train_scp, train_utt2label), SpoofDataset(validation_scp, validation_utt2label), SpoofDataset(evaluation_scp, evaluation_utt2label)

training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(validation_set, **params)
evaluation_generator = data.DataLoader(evaluation_set, **params)

validation_generator2 = data.DataLoader(validation_set, **params)

dev_utt2label  = ako.read_key_label(validation_utt2label)
eval_utt2label = ako.read_key_label(evaluation_utt2label)
gen_1, gen_2 = {}, {}
for epoch in range(max_epochs):
    for i_batch, sample_batched in enumerate(validation_generator):
        #key = sample_batched[0][0]
        #label = dev_utt2label[key]
        gen_1[i_batch] = sample_batched[1]
        if i_batch == 10: break 
    for i_batch, sample_batched in enumerate(validation_generator2):
        #key = sample_batched[0][0]
        #label = dev_utt2label[key]
        gen_2[i_batch] = sample_batched[1] 
        if i_batch == 10: break 

for i,value in gen_1.items():
    print((value==gen_2[i]).all()) 
        
