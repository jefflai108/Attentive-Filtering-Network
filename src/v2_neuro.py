from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.autograd import Variable
from torch import ones 

class ConvSmall(nn.Module):
    def __init__(self, input_size=(1,257,1091)):

        super(ConvSmall, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,2), dilation=(1,1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,2), dilation=(4,4)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,2), dilation=(8,8)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,2), dilation=(8,8)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
         
        ## Compute linear layer size
        self.flat_feats = self._get_flat_feats(input_size, self.features)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_feats, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        #self.apply(_weights_init)       
    
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
 
# ConvNet (utterance-based)
class ConvNet(nn.Module):
    def __init__(self, input_size=(1,257,1091)):

        super(ConvNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,2), dilation=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=(3,3), dilation=(8,8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=(3,3), dilation=(1,1)),
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
        
# ConvRes (utterance-based)
class ConvRes(nn.Module):
    def __init__(self, input_size=(1,257,1091)):

        super(ConvRes, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,2), dilation=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=(3,3), dilation=(8,8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=(3,3), dilation=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
         
        ## Compute linear layer size
        self.flat_feats = self._get_flat_feats(input_size, self.features)
        
        # Residual layers
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.re1 = nn.ReLU()
        self.ln2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.re2 = nn.ReLU()
        self.ln3 = nn.Linear(32, 32)
        self.dp1 = nn.Dropout(p=0.6)      
        self.bn3 = nn.BatchNorm1d(32)
        self.re3 = nn.ReLU()
        self.ln4 = nn.Linear(32, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.re4 = nn.ReLU()
        self.ln5 = nn.Linear(32, 32)
        self.dp2 = nn.Dropout(p=0.8)
        self.bn5 = nn.BatchNorm1d(32)
        self.re5 = nn.ReLU()
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

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
        x = self.ln1(flat_feats)
        residual = x
        x = self.ln3(self.re2(self.bn2(self.ln2(self.re1(self.bn1(x))))))
        x += residual
        x = self.dp1(x)
        residual = x
        x = self.ln5(self.re4(self.bn4(self.ln4(self.re3(self.bn3(x))))))
        x += residual
        x = self.dp2(x)
        out = self.sigmoid(self.ln6(self.re5(self.bn5(x))))
        return out
 
 # ResNet2 (utterance-based)
class ResNet2(nn.Module):
    def __init__(self, input_size=(1,257,1091)):

        super(ResNet2, self).__init__()
 
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,2), dilation=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU()
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU()
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        
        self.mp1  = nn.MaxPool2d(kernel_size=2)
        self.cnn4 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,2), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(16)
        self.re3  = nn.ReLU()
        self.cnn5 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        self.bn4  = nn.BatchNorm2d(16)
        self.re4  = nn.ReLU()
        self.cnn6 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        
        self.mp2  = nn.MaxPool2d(kernel_size=2)
        self.cnn7 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,2), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(16)
        self.re5  = nn.ReLU()
        self.cnn8 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        self.bn6  = nn.BatchNorm2d(16)
        self.re6  = nn.ReLU()
        self.cnn9 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        
        self.mp3  = nn.MaxPool2d(kernel_size=2)
        self.cnn10= nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,2), dilation=(8,8)) # no padding 
        
        # (N x 64 x 8 x 11) to (N x 64*8*11)
        self.flat_feats = 16*8*11
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU()
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU()
        self.ln3 = nn.Linear(32, 32)
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU()
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU()
        self.ln5 = nn.Linear(32, 32)
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU()
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                #kaiming_normal_(m.weight)
                xavier_normal_(m.weight)
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
        # feature
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual 
        x = self.cnn4(self.mp1(x))
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual 
        x = self.cnn7(self.mp2(x))
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual 
        x = self.cnn10(self.mp3(x))
        # classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        #x = F.dropout2d(x, training=True)
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        #x = F.dropout2d(x, training=True)
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        
        return out

 # ResNet (utterance-based)
class ResNet(nn.Module):
    def __init__(self, input_size=(1,257,1091)):

        super(ResNet, self).__init__()
 
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,2), dilation=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU()
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU()
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        
        self.mp1  = nn.MaxPool2d(kernel_size=2)
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,2), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU()
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU()
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        
        self.mp2  = nn.MaxPool2d(kernel_size=2)
        self.cnn7 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,2), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(64)
        self.re5  = nn.ReLU()
        self.cnn8 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        self.bn6  = nn.BatchNorm2d(64)
        self.re6  = nn.ReLU()
        self.cnn9 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
        
        self.mp3  = nn.MaxPool2d(kernel_size=2)
        self.cnn10= nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,2), dilation=(8,8)) # no padding 
        
        # (N x 64 x 8 x 11) to (N x 64*8*11)
        self.flat_feats = 64*8*11
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU()
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU()
        self.ln3 = nn.Linear(32, 32)
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU()
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU()
        self.ln5 = nn.Linear(32, 32)
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU()
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                #kaiming_normal_(m.weight)
                xavier_normal_(m.weight)
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
        # feature
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual 
        x = self.cnn4(self.mp1(x))
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual 
        x = self.cnn7(self.mp2(x))
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual 
        x = self.cnn10(self.mp3(x))
        # classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        #x = F.dropout2d(x, training=True)
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        #x = F.dropout2d(x, training=True)
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        
        return out

