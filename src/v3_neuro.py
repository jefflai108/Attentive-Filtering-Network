from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
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
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class network_9layers(nn.Module):
    def __init__(self, input_size=(1,257,1091)):
        
        super(network_9layers, self).__init__()
        
        self.features = nn.Sequential(
            mfm(1, 16, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=(2,3)), 
            group(16, 24, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=(2,3)),
            group(24, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,3)), 
            group(32, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,3)),
            group(16, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,3)),
            )

        self.flat_feats = self._get_flat_feats(input_size, self.features)
        
        self.fc1 = mfm(self.flat_feats, 64, type=0)
        self.fc2 = nn.Linear(64, 1)

        ## Weights initialization
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
 
    def _get_flat_feats(self, in_size, feats):
        f = feats(Variable(ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.flat_feats)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.sigmoid(x)

class network_9layers_v2(nn.Module):
    def __init__(self, input_size=(1,257,1091)):
        
        super(network_9layers_v2, self).__init__()
        
        self.features = nn.Sequential(
            mfm(1, 16, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=(2,3)), 
            group(16, 24, 3, 1, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=(2,3)),
            group(24, 32, 3, 1, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,3)), 
            group(32, 32, 3, 1, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,3)),
            group(32, 32, 3, 1, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=(2,3)),
            )

        self.flat_feats = self._get_flat_feats(input_size, self.features)
        
        self.fc1 = mfm(self.flat_feats, 32, type=0)
        self.fc2 = nn.Linear(32, 1)

    def _get_flat_feats(self, in_size, feats):
        f = feats(Variable(ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
 
    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(-1, self.flat_feats)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model

def LightCNN_9Layers_v2(**kwargs):
    model = network_9layers_v2(**kwargs)
    return model
