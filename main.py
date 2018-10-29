## Utilities
from __future__ import print_function
import argparse
import random
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
from src.v1_logger import setup_logs
from src.data_reader.v3_dataset import SpoofDataset
from src.v4_validation import validation
from src.v4_prediction import prediction
from src.v1_training import train, snapshot
from src.v3_neuro import LightCNN_9Layers
from src.v5_neuro import ResNet
from src.attention_neuro.residual_attention_network import ResidualAttentionModel
from src.attention_neuro.simple_attention_network import AttenResNet, PreAttenResNet, AttenResNet2, AttenResNet4, AttenResNet5
from src.attention_neuro.complex_attention_network import CAttenResNet1
from src.attention_neuro.recurrent_attention import BGRU, BLSTM
##############################################################
############ Control Center and Hyperparameter ###############
feat_dim = 257
M = 1091
select_best = 'eer' # eer or val
rnn = False # rnn
batch_size = test_batch_size = 4
atten_channel = 16
temperature = 2
atten_activation = 'sigmoid' 

def load_model(model, model_path, freeze=False):
    """load pre-trained model
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model params is', model_params)
    
    return model 

## v1_neuro
#run_name = "feed-forward" + time.strftime("-%Y-%m-%d_%H_%M")
#model = FeedForward(feat_dim*(2*M+1))
## v3_neuro
#run_name = "mfm" + time.strftime("-%Y-%m-%d_%H_%M")
#model = LightCNN_9Layers(input_size=(1,feat_dim,M))
## v5_neuro
#run_name = "conv-net" + time.strftime("-%Y-%m-%d_%H_%M")
#model = ResNet(input_size=(1,feat_dim,M))
# attention_neuro
run_name = "attention" + time.strftime("-%Y-%m-%d_%H_%M_%S")
#pretrain_path = '/export/b19/jlai/cstr/spoof/model/snapshots/attention/attention-2018-07-10_07_21_16-model_best.pth'
#pretrain = load_model(ResNet(), pretrain_path, freeze=False)
#model = PreAttenResNet(pretrain, atten_activation, atten_channel)
model = AttenResNet4(atten_activation, atten_channel)
#model = CAttenResNet1()
##############################################################

def main():
    ##############################################################
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-scp', required=True,
                        help='kaldi train scp file')
    parser.add_argument('--train-utt2label', required=True,
                        help='train utt2label')
    parser.add_argument('--validation-scp', required=True,
                        help='kaldi dev scp file')
    parser.add_argument('--validation-utt2label', required=True,
                        help='dev utt2label')
    parser.add_argument('--eval-scp',
                        help='kaldi eval scp file')
    parser.add_argument('--eval-utt2label',
                        help='train utt2label')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hidden-dim', type=int, default=100,
                        help='number of neurones in the hidden dimension')
    parser.add_argument('--plot-wd', help='training plot directory')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    #print('temperature is', temperature)

    # Global timer
    global_timer = timer()

    # Setup logs
    logger = setup_logs(args.logging_dir, run_name)

    # Setting random seeds for reproducibility.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True # CUDA determinism 
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    ##############################################################
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False,
              'worker_init_fn': np.random.seed(args.seed)} if use_cuda else {}

    logger.info('===> loading train and dev dataset')
    training_set   = SpoofDataset(args.train_scp, args.train_utt2label)
    validation_set = SpoofDataset(args.validation_scp, args.validation_utt2label)
    train_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **params) # set shuffle to True
    validation_loader = data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, **params) # set shuffle to False

    logger.info('===> loading eval dataset')
    eval_set = SpoofDataset(args.eval_scp, args.eval_utt2label)
    eval_loader = data.DataLoader(eval_set, batch_size=test_batch_size, shuffle=False, **params) # set shuffle to False
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ###########################################################
    ## Start training
    best_eer, best_loss = np.inf, np.inf
    early_stopping, max_patience = 0, 5 # early stopping and maximum patience
    print(run_name)
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        train(args, model, device, train_loader, optimizer, epoch, rnn)
        #train(args, model, device, train_loader, optimizer, epoch, args.train_scp, args.train_utt2label, args.plot_wd, rnn=False)
        val_loss, eer = validation(args, model, device, validation_loader, args.validation_scp, args.validation_utt2label, rnn)
        scheduler.step(val_loss)
        # Save
        if select_best == 'eer':
            is_best  = eer < best_eer
            best_eer = min(eer, best_eer)
        elif select_best == 'val':
            is_best  = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
        snapshot(args.logging_dir, run_name, is_best, {
                'epoch': epoch + 1,
                'best_eer': best_eer,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
        })
        # Early stopping 
        if is_best == 1: 
            early_stopping = 0 
        else: early_stopping += 1
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
        if early_stopping == max_patience: 
            break 
    ###########################################################
    ## Prediction
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(args.logging_dir,
                                        run_name + '-model_best.pth'
                                        )
                           )
    model.load_state_dict(checkpoint['state_dict'])

    eval_loss, eval_eer = prediction(args, model, device, eval_loader, args.eval_scp, args.eval_utt2label, rnn)
    ###########################################################
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    main()
