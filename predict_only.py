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

## Custom Imports
from src.data_reader.v3_dataset import SpoofDataset
from src.v1_logger import setup_logs
from src.v1_metrics import compute_eer
from src.v4_prediction import prediction, scores
from src.attention_neuro.simple_attention_network import AttenResNet, PreAttenResNet, AttenResNet2, AttenResNet3, AttenResNet4, AttenResNet5

run_name = "pred" + time.strftime("-%y-%m-%d_%h_%m")

feat_dim = 257
m = 1091
atten_channel = 16
atten_activation = 'sigmoid' 
temperature = 10 
#model = AttenResNet5(atten_activation, atten_channel, temperature)
model = AttenResNet2(atten_activation, atten_channel)

### trained model weights 
model_dir = '/export/b19/jlai/cstr/spoof/model/snapshots/attention/'
"""
## ResNet: remove one residual block in classifier 
## snapshots/scoring/resnet
model1 = model_dir + 'conv-net-2018-06-30_14_30-model_best.pth'
model2 = model_dir + 'conv-net-2018-06-30_14_31-model_best.pth'
model3 = model_dir + 'conv-net-2018-06-30_14_35-model_best.pth'
model4 = model_dir + 'conv-net-2018-06-30_14_36-model_best.pth'
model5 = model_dir + 'conv-net-2018-06-30_14_38-model_best.pth'
model6 = model_dir + 'conv-net-2018-06-30_14_40-model_best.pth'
models = [model1, model2, model3, model4, model5, model6]
# train: 0
# dev: 6.82
# eval: 10.92
## ResNet: add the 5th residual block in features 
## snapshots/scoring/resnet2
model1 = model_dir + 'conv-net-2018-06-29_15_28-model_best.pth'
model2 = model_dir + 'conv-net-2018-06-29_15_30-model_best.pth'
model3 = model_dir + 'conv-net-2018-06-29_15_40-model_best.pth'
model4 = model_dir + 'conv-net-2018-06-29_17_33-model_best.pth'
model5 = model_dir + 'conv-net-2018-06-29_19_37-model_best.pth'
model6 = model_dir + 'conv-net-2018-06-29_19_38-model_best.pth'
model7 = model_dir + 'conv-net-2018-06-30_07_09-model_best.pth'
model8 = model_dir + 'conv-net-2018-06-30_07_10-model_best.pth'
model9 = model_dir + 'conv-net-2018-06-29_18_08-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
# train: 0.00
# dev: 6.69
# eval: 10.30
## LCNN: modify channel to 16-24-32-16-16 and no dropout
## snapshots/scoring/lcnn
model1 = model_dir + 'mfm-2018-07-03_06_24-model_best.pth'
model2 = model_dir + 'mfm-2018-07-03_06_18-model_best.pth'
model3 = model_dir + 'mfm-2018-07-03_06_21-model_best.pth'
model4 = model_dir + 'mfm-2018-07-03_06_25-model_best.pth'
model5 = model_dir + 'mfm-2018-07-03_06_27-model_best.pth'
model6 = model_dir + 'mfm-2018-07-03_06_29-model_best.pth'
models = [model1, model2, model3, model4, model5, model6]
# train: 0.13
# dev: 6.47
# eval: 16.08 
## ResNet: lower lr to 0.0005 and replace relu with elu 
## snapshots/scoring/resnet3
model1 = model_dir + 'conv-net-2018-07-04_09_20-model_best.pth'
model2 = model_dir + 'conv-net-2018-07-04_09_22-model_best.pth'
model3 = model_dir + 'conv-net-2018-07-04_09_24-model_best.pth'
model4 = model_dir + 'conv-net-2018-07-04_11_48-model_best.pth'
model5 = model_dir + 'conv-net-2018-07-04_11_49-model_best.pth'
model6 = model_dir + 'conv-net-2018-07-04_11_50-model_best.pth'
model7 = model_dir + 'conv-net-2018-07-04_11_51-model_best.pth'
model8 = model_dir + 'conv-net-2018-07-04_11_52-model_best.pth'
model9 = model_dir + 'conv-net-2018-07-04_09_46-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
# train: 0.0
# dev: 7.49
# eval: 10.16
## ResNet4: longer skip connections
## snapshots/scoring/resnet4
model1 = model_dir + 'conv-net-2018-07-04_05_44-model_best.pth'
model2 = model_dir + 'conv-net-2018-07-04_05_46-model_best.pth'
model3 = model_dir + 'conv-net-2018-07-04_05_47-model_best.pth'
model4 = model_dir + 'conv-net-2018-07-04_05_48-model_best.pth'
model5 = model_dir + 'conv-net-2018-07-04_12_09-model_best.pth'
model6 = model_dir + 'conv-net-2018-07-04_12_11-model_best.pth'
model7 = model_dir + 'conv-net-2018-07-04_12_12-model_best.pth'
model8 = model_dir + 'conv-net-2018-07-04_20_35-model_best.pth'
model9 = model_dir + 'conv-net-2018-07-04_20_37-model_best.pth'
model10= model_dir + 'conv-net-2018-07-04_20_38-model_best.pth'
model11= model_dir + 'conv-net-2018-07-04_20_39-model_best.pth'
model12= model_dir + 'conv-net-2018-07-04_20_43-model_best.pth'
model13= model_dir + 'conv-net-2018-07-04_06_07-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13]
# train: 0.0
# dev: 6.36
# eval: 10.51
## AttenResNet (channel=16, sigmoid)
## snapshots/scoring/attention1
model1 = model_dir + 'attention-2018-07-11_15_24_48-model_best.pth'
model2 = model_dir + 'attention-2018-07-11_15_25_33-model_best.pth'
model3 = model_dir + 'attention-2018-07-11_15_25_44-model_best.pth'
model4 = model_dir + 'attention-2018-07-11_15_29_00-model_best.pth'
model5 = model_dir + 'attention-2018-07-17_08_48_24-model_best.pth'
model6 = model_dir + 'attention-2018-07-17_08_49_00-model_best.pth'
model7 = model_dir + 'attention-2018-07-17_08_50_24-model_best.pth'
model8 = model_dir + 'attention-2018-07-11_15_29_38-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train: 0.19
# dev: 6.6
# eval: 9.93
## AttenResNet2 (channel=16, sigmoid, attention residual)
## snapshots/scoring/attention2
model1 = model_dir + 'attention-2018-07-17_09_12_16-model_best.pth'
model2 = model_dir + 'attention-2018-07-17_09_13_10-model_best.pth'
model3 = model_dir + 'attention-2018-07-18_15_21_01-model_best.pth'
model4 = model_dir + 'attention-2018-07-18_15_19_31-model_best.pth'
model5 = model_dir + 'attention-2018-07-18_05_08_34-model_best.pth'
model6 = model_dir + 'attention-2018-07-18_15_19_34-model_best.pth'
model7 = model_dir + 'attention-2018-07-18_15_19_38-model_best.pth'
model8 = model_dir + 'attention-2018-07-17_09_13_56-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train: 0.13
# dev: 6.55
# eval: 8.99
## AttenResNet4 (channel=16, tanh, attention residual)
## snapshots/scoring/attention3
model1 = model_dir + 'attention-2018-07-19_07_07_22-model_best.pth'
model2 = model_dir + 'attention-2018-07-19_07_08_57-model_best.pth'
model3 = model_dir + 'attention-2018-07-19_07_24_01-model_best.pth'
model4 = model_dir + 'attention-2018-07-19_07_24_19-model_best.pth'
model5 = model_dir + 'attention-2018-07-19_16_02_01-model_best.pth'
model6 = model_dir + 'attention-2018-07-19_21_59_11-model_best.pth'
model7 = model_dir + 'attention-2018-07-19_16_02_21-model_best.pth'
model8 = model_dir + 'attention-2018-07-19_07_40_03-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train: 0.11
# dev: 6.87
# eval: 10.17
## AttenResNet4 (channel=16, softmax(dim=2), attention residual)
## snapshots/scoring/attention4
model1 = model_dir + 'attention-2018-07-19_16_10_54-model_best.pth'
model2 = model_dir + 'attention-2018-07-19_22_50_10-model_best.pth'
model3 = model_dir + 'attention-2018-07-19_16_11_46-model_best.pth'
model4 = model_dir + 'attention-2018-07-19_16_11_50-model_best.pth'
model5 = model_dir + 'attention-2018-07-20_11_57_44-model_best.pth'
model6 = model_dir + 'attention-2018-07-20_11_58_46-model_best.pth'
model7 = model_dir + 'attention-2018-07-19_16_12_32-model_best.pth'
model8 = model_dir + 'attention-2018-07-19_23_40_21-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train: 0.1
# dev: 6.52
# eval: 9.34
## AttenResNet4 (channel=16, softmax(dim=3), attention residual)
## snapshots/scoring/attention5
model1 = model_dir + 'attention-2018-07-19_17_31_12-model_best.pth'
model2 = model_dir + 'attention-2018-07-19_18_30_58-model_best.pth'
model3 = model_dir + 'attention-2018-07-20_05_20_07-model_best.pth'
model4 = model_dir + 'attention-2018-07-20_16_40_54-model_best.pth'
model5 = model_dir + 'attention-2018-07-19_18_26_32-model_best.pth'
model6 = model_dir + 'attention-2018-07-19_18_31_09-model_best.pth'
model7 = model_dir + 'attention-2018-07-19_20_48_59-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7]
# train: 0.13
# dev: 6.62
# eval: 9.28
## AttenResNet5 (channel=16, softmax(dim=3,T=10), attention residual)
## snapshots/scoring/attention6
model1 = model_dir + 'attention-2018-07-20_17_55_06-model_best.pth'
model2 = model_dir + 'attention-2018-07-20_17_54_41-model_best.pth'
model3 = model_dir + 'attention-2018-07-20_17_54_54-model_best.pth'
model4 = model_dir + 'attention-2018-07-20_17_55_11-model_best.pth'
model5 = model_dir + 'attention-2018-07-21_14_56_24-model_best.pth'
model6 = model_dir + 'attention-2018-07-21_14_52_40-model_best.pth'
model7 = model_dir + 'attention-2018-07-21_14_56_41-model_best.pth'
model8 = model_dir + 'attention-2018-07-20_18_00_27-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train: 0.15
# dev: 6.36
# eval: 9.7
## AttenResNet5 (channel=16, softmax(dim=2,T=10), attention residual)
## snapshots/scoring/attention7
model1 = model_dir + 'attention-2018-07-21_18_47_35-model_best.pth'
model2 = model_dir + 'attention-2018-07-21_19_39_14-model_best.pth'
model3 = model_dir + 'attention-2018-07-22_11_27_26-model_best.pth'
model4 = model_dir + 'attention-2018-07-22_11_27_34-model_best.pth'
model5 = model_dir + 'attention-2018-07-22_11_25_38-model_best.pth'
model6 = model_dir + 'attention-2018-07-21_18_47_39-model_best.pth'
model7 = model_dir + 'attention-2018-07-21_18_50_09-model_best.pth'
model8 = model_dir + 'attention-2018-07-21_18_51_34-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train: 0.2
# dev: 6.48
# eval: 9.96
## AttenResNet5 (channel=16, softmax(dim=2,T=10), attention residual)
## snapshots/scoring/
model1 = model_dir + '-model_best.pth'
model2 = model_dir + '-model_best.pth'
model3 = model_dir + '-model_best.pth'
model4 = model_dir + '-model_best.pth'
model5 = model_dir + '-model_best.pth'
model6 = model_dir + '-model_best.pth'
model7 = model_dir + '-model_best.pth'
model8 = model_dir + '-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8]
# train:
# dev: 
# eval: 
"""
## AttenResNet2 (channel=16, sigmoid, attention residual)
## snapshots/scoring/attention8
model1 = model_dir + 'attention-2018-07-17_09_12_16-model_best.pth'
model2 = model_dir + 'attention-2018-07-17_09_13_10-model_best.pth'
model3 = model_dir + 'attention-2018-07-18_15_21_01-model_best.pth'
model4 = model_dir + 'attention-2018-07-18_15_19_31-model_best.pth'
model5 = model_dir + 'attention-2018-07-18_05_08_34-model_best.pth'
model6 = model_dir + 'attention-2018-07-18_15_19_34-model_best.pth'
model7 = model_dir + 'attention-2018-07-18_15_19_38-model_best.pth'
model8 = model_dir + 'attention-2018-07-17_09_13_56-model_best.pth'
model9 = model_dir + 'attention-2018-07-25_16_12_12-model_best.pth'
model10 = model_dir + 'attention-2018-07-25_16_12_21-model_best.pth'
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
# train: 
# dev: 
# eval: 

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
    parser.add_argument('--scoring-txt', 
                        help='output scoring text file')
    parser.add_argument('--label-txt', 
                        help='output labels text file')
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
    """ 
    ###################### for single model #####################
    logger.info('===> loading {} for prediction'.format(args.model_path))
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model params is', model_params)

    eval_loss, eval_eer = prediction(args, model, device, eval_loader, args.eval_scp, args.eval_utt2label)
    """
    ################### for multiple models #####################
    np.set_printoptions(threshold=np.nan)
    sum_preds = 0
    for model_i in models: 
        logger.info('===> loading {} for prediction'.format(model_i))
        checkpoint = torch.load(model_i, map_location=lambda storage, loc: storage) # load everything onto CPU
        model.load_state_dict(checkpoint['state_dict'])
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model params is', model_params)

        eval_preds, eval_labels = scores(args, model, device, eval_loader, args.eval_scp, args.eval_utt2label)
        sum_preds += eval_preds
    sum_preds /= len(models)  # get the average
    eval_eer = compute_eer(eval_labels, sum_preds)
    np.savetxt(args.scoring_txt, sum_preds)
    np.savetxt(args.label_txt, eval_labels)
    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> evalidation set: EER: {:.4f}\n'.format(eval_eer))
    ###########################################################
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    main()
