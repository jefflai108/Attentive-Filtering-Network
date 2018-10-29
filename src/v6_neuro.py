from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.autograd import Variable
from torch import ones 

# ResNet (utterance-based) for cqcc spectrogram 
class ResNet(nn.Module):
    def __init__(self, input_size=(1,863,1278)):

        super(ResNet, self).__init__()
 
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding 
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding 
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 16, kernel_size=(3,3), dilation=(8,8)) # no padding 
        

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 16*26*12
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
    
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual 
        x = self.cnn4(self.mp1(x))
        #print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual 
        x = self.cnn7(self.mp2(x))
        #print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual 
        x = self.cnn10(self.mp3(x))
        #print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual 
        x = self.cnn13(self.mp4(x))
        #print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual 
        x = self.cnn16(self.mp5(x))
        #print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        #print(out.size()) 
        
        return out

