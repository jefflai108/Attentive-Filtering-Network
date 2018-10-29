from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.autograd import Variable
from torch import ones 

# Fully convolutoin for log-fbank
class FConv(nn.Module):
    def __init__(self):

        super(FConv, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(4,4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(8,8)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fc = nn.Linear(32,1)

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU):
                xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
 
    def forward(self, inputs):
        batch_size = inputs.size(0) # input size is N*C*H*W
        c = self.convolution(inputs) 
        # suppose c is your feature map with size N*C*H*W
        c = torch.mean(c.view(c.size(0), c.size(1), -1), dim=2) # now c is of size N*C
        c = self.fc(c)
        return F.sigmoid(c)

