from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_layers import ResidualBlock, L1Penalty

class AttentionModule_stg0(nn.Module):
    """
    attention module with softmax branch and trunk branch, Residual Attention Network CVPR 2017
                   --(trunk)-->RB(d=1)-->RB(d=1)-------------------------------------------------------------------------------------------------------------------------|
                  /                                                                                                                                                      |
    x -->RB(d=1)-|                                                                                                                                                       |
                  \                                                                                                                                                      |                      
                   --(softmax)-->mp-->RB(d=2)-|-->mp-->RB(d=4)-|-->mp-->RB(d=8)-|-->mp-->RB(d=16)-->RB(d=1)-->up-+-->RB(d=1)-->up-+-->RB(d=1)-->up-+-->RB(d=1)-->up--|   |
                                              |                |                |--------------RB(d=1)-----------|                |                |                 |   |
                                              |                |-------------------------------RB(d=1)----------------------------|                |                 |   |
                                              |------------------------------------------------RB(d=1)---------------------------------------------|                 |   |
                                                                                                                                                                     |   |
                                                                                                                                   |---sigmoid<--conv1*1<--conv1*1<--|   |
                                                                                                             out<--RB(d=1)<--+--<--*                                     |
                                                                                                                             |-----|-------------------------------------|
    """
    def __init__(self, in_channels, out_channels, size1=(128,545), size2=(120,529), size3=(104,497), size4=(72,186), l1weight=0.2):
            
        super(AttentionModule_stg0, self).__init__()
        self.l1weight = l1weight 
        self.pre = ResidualBlock(in_channels, 1)

        ## trunk branch 
        self.trunk = nn.Sequential(
            ResidualBlock(in_channels, 1),
            ResidualBlock(in_channels, 1)
        )
        ## softmax branch: bottom-up 
        self.mp1   = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.sm1   = ResidualBlock(in_channels, (4,8))
        self.skip1 = ResidualBlock(in_channels, 1)
        
        self.mp2   = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.sm2   = ResidualBlock(in_channels, (8,16))
        self.skip2 = ResidualBlock(in_channels, 1)
        
        self.mp3   = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.sm3   = ResidualBlock(in_channels, (16,32))
        self.skip3 = ResidualBlock(in_channels, 1)
        
        self.mp4   = nn.MaxPool2d(kernel_size=3, stride=(2,2))
        self.sm4   = nn.Sequential(
            ResidualBlock(in_channels, (16,32)),
            ResidualBlock(in_channels, 1)
        )
        ## softmax branch: top-down 
        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.sm5   = ResidualBlock(in_channels, 1)
        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.sm6   = ResidualBlock(in_channels, 1)
        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.sm7   = ResidualBlock(in_channels, 1)
        self.up1   = nn.UpsamplingBilinear2d(size=size1)
        # 1*1 convolution blocks 
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels , kernel_size=1, stride=1, bias=False),
            #nn.Sigmoid()
            nn.Softmax2d()
        )
        
        self.post = ResidualBlock(in_channels, 1)

    def forward(self, x):
        #print('attention!')
        x = self.pre(x)
        #print('pre', x.size())
        out_trunk = self.trunk(x)
        #print('trunk', out_trunk.size()) 
        out_mp1 = self.mp1(x)
        #print('mp1', out_mp1.size()) 
        out_sm1 = self.sm1(out_mp1)
        #print('sm1', out_sm1.size()) 
        out_skip1 = self.skip1(out_sm1)
        #print('skip1', out_skip1.size())
        out_mp2 = self.mp2(out_sm1)
        #print('mp2', out_mp2.size()) 
        out_sm2 = self.sm2(out_mp2)
        #print('sm2', out_sm2.size())
        out_skip2 = self.skip2(out_sm2)
        #print('skip2', out_skip2.size())
        out_mp3 = self.mp3(out_sm2)
        #print('mp3', out_mp3.size()) 
        out_sm3 = self.sm3(out_mp3)
        #print('sm3', out_sm3.size())
        out_skip3 = self.skip3(out_sm3)
        #print('skip3', out_skip3.size())
        out_mp4 = self.mp4(out_sm3)
        #print('mp4', out_mp4.size()) 
        out_sm4 = self.sm4(out_mp4)
        #print('sm4', out_sm4.size())
        out_up4 = self.up4(out_sm4) 
        #print('up4', out_up4.size())
        out = out_up4 + out_skip3
        #print('out', out.size()) 
        out_sm5 = self.sm5(out)
        #print('sm5', out_sm5.size())
        out_up3 = self.up3(out_sm5) 
        #print('up3', out_up3.size())
        out = out_up3 + out_skip2
        #print('out', out.size()) 
        out_sm6 = self.sm6(out)
        #print('sm6', out_sm6.size())
        out_up2 = self.up2(out_sm6) 
        #print('up2', out_up2.size())
        out = out_up2 + out_skip1
        #print('out', out.size()) 
        out_sm7 = self.sm7(out)
        #print('sm7', out_sm7.size())
        out_up1 = self.up1(out_sm7) 
        #print('up1', out_up1.size())
        out_conv1 = self.conv1(out_up1)
        #print('conv1', out_conv1.size()) 
        #out = (out_conv1) * out_trunk
        out = (1 + out_conv1) * out_trunk
        #print('out', out.size()) 
        out_post = self.post(out)
        #print('post', out_post.size())
        
        return out_post

