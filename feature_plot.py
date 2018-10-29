## Utilities
from __future__ import print_function
import argparse
import time
import os
import logging
from timeit import default_timer as timer

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

## Custrom Imports
from src.data_reader.v3_dataset import SpoofDataset
from src.v1_logger import setup_logs
from src.v4_plot import retrieve_weight
from src.attention_neuro.simple_attention_network import AttenResNet, PreAttenResNet, AttenResNet2, AttenResNet4, AttenResNet5

run_name = "pred" + time.strftime("-%y-%m-%d_%h_%m")

feat_dim = 257
m = 1091
rnn = False # rnn
atten_channel = 16
atten_activation = 'sigmoid'
temperature = 1

#model = AttenResNet5(atten_activation, atten_channel, temperature)
model = AttenResNet4(atten_activation, atten_channel)
#model = AttenResNet(atten_activation, atten_channel)
model_dir = '/export/b19/jlai/cstr/spoof/model/snapshots/attention/'
#model1 = model_dir + 'attention-2018-07-11_15_25_44-model_best.pth' # AttenResnet1, c=16, sigmoid 
#model1 = model_dir + 'attention-2018-07-10_16_15_25-model_best.pth' # AttenResnet1, c=1, softmax 
model1 = model_dir + 'attention-2018-07-17_09_13_56-model_best.pth' # AttenResnet2, c=16, sigmoid, attention residual 
#model1 = model_dir + 'attention-2018-07-19_21_59_11-model_best.pth' # AttenResnet4, c=16, tanh, attention residual 
#model1 = model_dir + 'attention-2018-07-19_16_11_46-model_best.pth' # AttenResnet4, c=16, softmax2, attention residual
#model1 = model_dir + 'attention-2018-07-19_20_48_59-model_best.pth' # AttenResnet4, c=16, softmax3, attention residual
#model1 = model_dir + 'attention-2018-07-20_17_55_06-model_best.pth' # AttenResnet5, c=16, softmax3, T=10, attention residual
#model1 = model_dir + 'attention-2018-07-21_07_07_15-model_best.pth' # AttenResnet5, c=16, softmax3, T=100, attention residual
#model1 = model_dir + 'attention-2018-07-21_13_14_42-model_best.pth' # AttenResnet5, c=16, softmax3, T=0.1, attention residual
#model1 = model_dir + 'attention-2018-07-20_19_06_27-model_best.pth' # AttenResnet5, c=16, softmax3, T=5, attention residual
#model1 = model_dir + 'attention-2018-07-21_07_02_09-model_best.pth' # AttenResnet5, c=16, softmax3, T=20, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax2, T=10, attention residual
#model1 = model_dir + 'attention-2018-07-23_17_30_53-model_best.pth' # AttenResnet5, c=16, softmax3, T=0.01, attention residual
#model1 = model_dir + 'attention-2018-07-23_18_01_13-model_best.pth' # AttenResnet5, c=16, softmax3, T=0.05, attention residual
#model1 = model_dir + 'attention-2018-07-23_23_45_07-model_best.pth' # AttenResnet5, c=16, softmax3, T=0.001, attention residual
#model1 = model_dir + 'attention-2018-07-24_01_38_24-model_best.pth' # AttenResnet5, c=16, softmax3, T=0.2, attention residual
#model1 = model_dir + 'attention-2018-07-24_03_40_28-model_best.pth' # AttenResnet5, c=16, softmax3, T=1000, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=2, attention residual
#model1 = model_dir + 'attention-2018-07-24_07_54_07-model_best.pth' # AttenResnet5, c=16, softmax3, T=0.5, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=3, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=4, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=6, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=7, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=8, attention residual
#model1 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth' # AttenResnet5, c=16, softmax3, T=9, attention residual
models = [model1]

def main():
    ##############################################################
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--eval-scp',
                        help='kaldi eval scp file')
    parser.add_argument('--eval-utt2label',
                        help='train utt2label')
    parser.add_argument('--model-path',
                        help='trained model')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--plot-dir', 
                        help='directory to save plots')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)

    # Global timer
    global_timer = timer()

    # Setup logs
    logger = setup_logs(args.logging_dir, run_name)

    # Setting random seeds for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    ##############################################################
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logger.info('===> loading eval dataset')
    eval_set = SpoofDataset(args.eval_scp, args.eval_utt2label)
    eval_loader = data.DataLoader(eval_set, batch_size=args.test_batch_size, shuffle=False, **params) # set shuffle to False
    ################### for multiple models #####################
    np.set_printoptions(threshold=np.nan)
    sum_preds = 0
    for model_i in models: 
        logger.info('===> loading {} for prediction'.format(model_i))
        checkpoint = torch.load(model_i)
        model.load_state_dict(checkpoint['state_dict'])
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model params is', model_params)

        retrieve_weight(args, model, device, eval_loader, args.eval_scp, args.eval_utt2label, args.plot_dir, rnn)
    logger.info("===> Final predictions done. Here is a snippet")
    ###########################################################
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    main()
