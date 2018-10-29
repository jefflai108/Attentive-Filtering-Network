from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class BLSTM(nn.Module):
    def __init__(self, batch=32, bidirectional=True):

        super(BLSTM, self).__init__()
        
        self.bidirectional = bidirectional
        self.hidden = self.init_hidden(batch)
        self.lstm = nn.LSTM(257, 50, num_layers=2, bidirectional=True)  
        self.fc = nn.Linear(50*2,1)

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
 
    def init_hidden(self, batch):
        return (torch.zeros(2*2, batch, 50).cuda(),torch.zeros(2*2, batch, 50).cuda()) 

    def forward(self, x):
        # input shiddenape is N*C*H*W, C is 1
        x = x.squeeze() # get rid of C
        x = x.transpose(1,2).transpose(0,1) # make it W*N*H
        output_seq,_ = self.lstm(x, self.hidden)
        #print(output_seq.size())
        r = output_seq[-1]
        #print(r.size())
        r = self.fc(r)

        return F.sigmoid(r)

class BGRU(nn.Module):
    def __init__(self, bidirectional=True):

        super(BGRU, self).__init__()
        
        self.bidirectional = bidirectional

        if bidirectional == True:
            self.gru = nn.GRU(257,10,num_layers=2,bidirectional=True)  
        else:
            self.gru = nn.GRU(257,10,num_layers=2,bidirectional=False)
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(3,10), stride=1, padding=1, dilation=(2,16)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(3,10), stride=1, padding=1, dilation=(4,32)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(3,10), stride=1, padding=1, dilation=(4,64))
        )

        self.avg = nn.Sequential( # dimension reduction
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(3,20), stride=(2,2))
        )
        self.fc = nn.Linear(70,1)
        
         ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU):
                init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
 
    def forward(self, x, hidden):
        # input shiddenape is N*C*H*W, C is 1
        x = x.squeeze() # get rid of C
        x = x.transpose(1,2).transpose(0,1) # make it W*N*H
        r, hidden = self.gru(x, hidden)
        r = r.transpose(1,0).transpose(2,1) # make it N*H*W
        r = r.contiguous()
        r = r.unsqueeze(1) # make it N*C*H*W, C is 1
        #print(r.size())        
        r = self.conv(r)
        #print(r.size())
        r = self.avg(r)
        #print(r.size())
        r = r.view(r.size(0), -1)
        #print(r.size())
        r = self.fc(r)

        return F.sigmoid(r), hidden

