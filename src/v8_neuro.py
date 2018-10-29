from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.autograd import Variable
from torch import ones 
import math

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1, dilation=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0, dilation)
        self.dp_a = nn.Dropout2d(p=0.5)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.dp   = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.dp_a(x)
        x = self.conv(x)
        x = self.dp(x)
        return x

class network_9layers(nn.Module):
    def __init__(self, input_1=(1,257,1091), input_2=(1,129,1091)):
        
        super(network_9layers, self).__init__()
        
        self.features1 = nn.Sequential(
            mfm(1, 16, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=(2,2)), 
            group(16, 24, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            group(24, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)), 
            group(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            group(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            )
        self.features2 = nn.Sequential(
            mfm(1, 16, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=(2,2)), 
            group(16, 24, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            group(24, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)), 
            group(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            group(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            )

        self.flat_feat1 = self._get_flat_feats(input_1, self.features1)
        self.flat_feat2 = self._get_flat_feats(input_2, self.features2)

        self.fc1a = mfm(self.flat_feat1, 32, type=0)
        self.fc1b = mfm(self.flat_feat2, 32, type=0)
        self.fc2 = mfm(64, 32, type=0)
        self.fc3 = nn.Linear(32, 1)

    def _get_flat_feats(self, in_size, feats):
        f = feats(Variable(ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
 
    def forward(self, x1, x2):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x1 = x1.view(-1, self.flat_feat1)
        x2 = x2.view(-1, self.flat_feat2)
        x1 = self.fc1a(x1)
        x2 = self.fc1b(x2)
        x1 = F.dropout2d(x1, p=0.5)
        x2 = F.dropout2d(x2, p=0.5)
        x3 = torch.cat((x1,x2),dim=1)
        x3 = self.fc2(x3)
        x3 = F.dropout(x3, p=0.5)
        x3 = self.fc3(x3)
        return F.sigmoid(x3)

def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model
