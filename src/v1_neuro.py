from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.autograd import Variable
from torch import ones 

# ConvNet
class ConvNet(nn.Module):
    def __init__(self, input_size=(1,257,1091)):

        super(ConvNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=(3,3), dilation=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
         
        ## Compute linear layer size
        self.flat_feats = self._get_flat_feats(input_size, self.features)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_feats, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
    
    def _get_flat_feats(self, in_size, feats):
        f = feats(Variable(ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
            
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        feats = self.features(x)
        flat_feats = feats.view(-1, self.flat_feats)
        #print(flat_feats.size())
        out = self.classifier(flat_feats)
        return out

# feed forward DNN
class FeedForward(nn.Module):
    def __init__(self, input_dim):

        super(FeedForward, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.AlphaDropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.classifier(x)
        return out

# CNN + GRU
class ConvGRU(nn.Module):
    def __init__(self):

        super(ConvGRU, self).__init__()

        self.conv1  = nn.Conv2d(1, 16, kernel_size=(5,5), padding=(2,2))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1  = nn.ReLU()
        self.conv2  = nn.Conv2d(16, 32, kernel_size=(5,5), padding=(2,2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2  = nn.ReLU()
        self.conv3  = nn.Conv2d(32, 64, kernel_size=(5,5), padding=(2,2))
        self.batch3 = nn.BatchNorm2d(64)
        self.relu3  = nn.ReLU()
        self.gru    = nn.GRU(25700, 1, bidirectional=True)
        self.fc1    = nn.Linear(512, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.drop1  = nn.Dropout(0.5)
        self.relu4  = nn.ReLU()
        self.fc2    = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.drop2  = nn.Dropout(0.5)
        self.relu5  = nn.ReLU()
        self.fc3    = nn.Linear(128, 2)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(0) # batch size is 1
        # input size is (batch_size x H x W x C)
        # reshape if to (batch_size x C x H x W) for CNN
        inputs = inputs.transpose(2,3).transpose(1,2)

        # Run through Conv2d, BatchNorm2d, ReLU layers
        h = self.conv1(inputs)
        h = self.batch1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.batch2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.batch3(h)
        h = self.relu3(h)

        h = h.squeeze() # get ride of the batc_size dim
        h = h.view(h.size(0), -1, 4).transpose(1,2) # reshape (C x H x W)

        r, hidden = self.gru(h, hidden) # BGRU unit is applied to each channel of CNN's output
        r = r.view(1, -1)

        f = self.fc1(r)
        f = self.drop1(f)
        f = self.relu4(f)
        f = self.fc2(f)
        f = self.drop2(f)
        f = self.relu5(f)
        f = self.fc3(f)
        return F.log_softmax(r, dim=1), hidden

