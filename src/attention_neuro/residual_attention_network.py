from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import numpy as np
from basic_layers import ResidualBlock, CRResidualBlock 
from attention_module import AttentionModule_stg0

class ResidualAttentionModel(nn.Module):
    def __init__(self):

        super(ResidualAttentionModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.rb1 = ResidualBlock(32, 1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2)
        self.features = nn.Sequential(
            AttentionModule_stg0(32, 32)
        )
        self.classifier = nn.Sequential( # dimension reduction
            CRResidualBlock(32, 8, (4,16)), 
            CRResidualBlock(8, 4, (8,32)), 
            CRResidualBlock(4, 2, (16,64)),
            CRResidualBlock(2, 1, (32,128))
        )
        self.mpool2 = nn.Sequential( # dimension reduction
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(3,20), stride=2)
        )
        self.fc = nn.Linear(189,1)
        
        ## Weights initialization
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                xavier_normal_(m.weight)
            elif classname.find('Linear') != -1:
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
 
    def forward(self, x):
        #print('1:', x.size()) 
        x = self.conv1(x)
        #print('2:', x.size()) 
        x = self.rb1(x)
        #print('3:', x.size()) 
        x = self.mpool1(x) 
        #print('4:', x.size())
        x = self.features(x)
        #print('5:', x.size())
        x = self.classifier(x)
        #print('6:', x.size())
        x = self.mpool2(x)
        #print('7:', x.size())
        x = x.view(x.size(0), -1)
        #print('8:', x.size())
        x = self.fc(x)

        return F.sigmoid(x)

