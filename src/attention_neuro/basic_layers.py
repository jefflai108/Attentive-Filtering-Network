from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.autograd import Function

## for complex_attention_network
class LocalAttenBlock(nn.Module):
    def __init__(self, in_channels, dilation=1):
        
        super(LocalAttenBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), dilation=dilation)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), dilation=1)
        self.bn2   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

## for simple_attention_network
class PlainConvBlock(nn.Module):
    def __init__(self, in_channels, dilation=1):
        
        super(PlainConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), dilation=dilation)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), dilation=1)
        self.bn2   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class ResidualBlock(nn.Module):
    """residual block (reference v5_neuro.ResNet)
    |-->bn2d-->relu-->conv2d-->bn2d-->relu-->conv2d--|
    x -----------------------------------------------+-->bn2d-->relu-->dilated_conv2d-->out
    """
    def __init__(self, in_channels, dilation=1):
        
        super(ResidualBlock, self).__init__()
        
        self.dilation = dilation
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn2   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn3   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), dilation=dilation, bias=False)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual 
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x

class CRResidualBlock(nn.Module):
    """channel-reduction-residual block (reference v5_neuro.ResNet)
    |-->bn2d-->relu-->conv2d-->bn2d-->relu-->conv2d--|
    x -----------------------------------------------+-->bn2d-->relu-->dilated_conv2d-->out
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        
        super(CRResidualBlock, self).__init__()
        
        self.dilation = dilation
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn2   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), bias=False)
        self.bn3   = nn.BatchNorm2d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), dilation=dilation, bias=False)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual 
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x

class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input, None

