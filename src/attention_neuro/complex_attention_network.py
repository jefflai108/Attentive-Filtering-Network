from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
from basic_layers import PlainConvBlock as PCB

# Complex-Attention-ResNet1: resnet with 1. attentive-filtering network 2. attentive-scoring network 
class CAttenResNet1(nn.Module):
    def __init__(self, atten_width=1, atten_channel=16, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851)):

        super(CAttenResNet1, self).__init__()

        ## attentive-filtering network: channel-expansion
        self.channel_expansion = nn.Sequential(
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        
        ## attentive-filtering network: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(4,8)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(8,16)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(16,32)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(32,64)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(64,128)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attentive-filtering network: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)
        
        ## attentive-filtering network: 1*1 conv channel-compression 
        self.channel_compression = nn.Sequential( 
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel/4, kernel_size=1, stride=1),
            nn.BatchNorm2d(atten_channel/4),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel/4, 1, kernel_size=1, stride=1)
        )

        ## attentive-filtering network: activation function 
        self.soft  = nn.Sigmoid()

        ## resnet 
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))

        ## resnet: residual-block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding

        ## resnet: residual-block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding

        ## resnet: residual-block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding 
        """
        ## resnet: attentive-scoring 3
        self.conv3  = nn.Sequential(
            nn.Conv2d(32, 32*atten_width, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.reduc3 = nn.Sequential(
            nn.Conv2d(32*atten_width, 32*atten_width, kernel_size=(6,8), stride=1, padding=(1,2), dilation=(23,18)),
            nn.BatchNorm2d(32*atten_width),
            nn.ReLU(inplace=True)
        )
        self.ffeat3 = 32*atten_width*1*1
        self.satt5  = nn.Sequential(
            nn.Linear(self.ffeat3, 1),
            nn.Sigmoid()
        )
        self.satt6  = nn.Sequential(
            nn.Linear(self.ffeat3, 1),
            nn.Tanh()
        )
        """
        ## resnet: residual-block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding 
        """
        ## resnet: attentive-scoring 4
        self.conv4  = nn.Sequential(
            nn.Conv2d(32, 32*atten_width, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.reduc4 = nn.Sequential(
            nn.Conv2d(32*atten_width, 32*atten_width, kernel_size=(5,5), stride=1, padding=(0,0), dilation=(10,11)),
            nn.BatchNorm2d(32*atten_width),
            nn.ReLU(inplace=True)
        )
        self.ffeat4 = 32*atten_width*1*1
        self.satt7  = nn.Sequential(
            nn.Linear(self.ffeat4, 1),
            nn.Sigmoid()
        )
        self.satt8  = nn.Sequential(
            nn.Linear(self.ffeat4, 1),
            nn.Tanh()
        )
        """
        ## resnet: residual-block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding 

        ## resnet: attentive-scoring 5
        self.conv5  = nn.Sequential(
            nn.Conv2d(32, 32*atten_width, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.reduc5 = nn.Sequential(
            nn.Conv2d(32*atten_width, 32*atten_width, kernel_size=(4,6), stride=1, padding=(0,0), dilation=1),
            nn.BatchNorm2d(32*atten_width),
            nn.ReLU(inplace=True)
        )
        self.ffeat5 = 32*atten_width*1*1
        self.satt9  = nn.Sequential(
            nn.Linear(self.ffeat5, 1),
            nn.Sigmoid()
        )
        self.satt10 = nn.Sequential(
            nn.Linear(self.ffeat5, 1),
            nn.Tanh()
        )

        ## resnet: scoring attention gate
        self.gate = nn.Softmax(dim=1)

        ## weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
    
    def forward(self, x):
        #print('size1', x.size())
        residual = x
        x = self.att1(self.down1(self.channel_expansion(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())    
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        #print('size5', x.size())
        x = self.att5(self.down5(x))
        #print(x.size())
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.channel_compression(self.up1(x))
        weight = self.soft(x)
        #print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual 
        
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual 
        x = self.cnn4(self.mp1(x))
        #print(x.size()) # (4, 32, 253, 541)
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual 
        x = self.cnn7(self.mp2(x))
        #print(x.size()) # (4, 32, 245, 262)
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual 
        x = self.cnn10(self.mp3(x))
        #print(x.size()) # (4, 32, 114, 123)
        """
        residual = x
        weight = self.conv3(x)
        s = (1 + weight) * residual
        s = self.reduc3(s)
        #print(s.size())
        s = s.view(-1, self.ffeat3)
        #print(s.size())
        score3 = self.satt5(s)
        conf3  = self.satt6(s)
        """
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual 
        x = self.cnn13(self.mp4(x))
        #print(x.size()) # (4, 32, 41, 45) 
        """
        residual = x
        weight = self.conv4(x)
        s = (1 + weight) * residual
        s = self.reduc4(s)
        #print(s.size())
        s = s.view(-1, self.ffeat4)
        #print(s.size())
        score4 = self.satt7(s)
        conf4  = self.satt8(s)
        """
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual 
        x = self.cnn16(self.mp5(x))
        #print(x.size()) # (4, 32, 4, 6)
        residual = x
        weight = self.conv5(x)
        s = (1 + weight) * residual
        s = self.reduc5(s)
        s = s.view(-1, self.ffeat5)
        score5 = self.satt9(s)
        conf5  = self.satt10(s)

        conf  = self.gate(conf5)
        out   = score5 * conf.view(-1,1)
        #conf = self.gate(torch.cat((conf4, conf5), dim=1))
        #out  = score4 * conf[:,0].view(-1,1) + score5 * conf[:,1].view(-1,1)
        #conf = self.gate(torch.cat((conf3, conf4, conf5), dim=1))
        #out  = score3 * conf[:,0].view(-1,1) + score4 * conf[:,1].view(-1,1) + score5 * conf[:,2].view(-1,1) 
        return out

