from __future__ import print_function 
import torch 
import torch.nn as nn
import numpy as np 

def main():    
    a=nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,2), dilation=(1,1))
    b=nn.BatchNorm2d(16)
    c=nn.ReLU()
    e=nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    f=nn.BatchNorm2d(16)
    g=nn.ReLU()
    h=nn.MaxPool2d(kernel_size=2)
    i=nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,2), dilation=(4,4))
    A=nn.BatchNorm2d(32)
    B=nn.ReLU()
    C=nn.MaxPool2d(kernel_size=2)
    D=nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,2), dilation=(8,8))
    E=nn.BatchNorm2d(32)
    F=nn.ReLU()
    G=nn.MaxPool2d(kernel_size=2)
    H=nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,2), dilation=(8,8))
    I=nn.BatchNorm2d(32)
    J=nn.ReLU()
    K=nn.MaxPool2d(kernel_size=2)
    
    input = torch.Tensor(32,1,257,1091)
    x = a(input)
    x = K(J(I(H(G(F(E(D(C(B(A(i(h(g(f(e(c(b(x))))))))))))))))))
    print(x.size())

    """
    cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2,3), dilation=(2,2))
    ## block 1
    bn1  = nn.BatchNorm2d(16)
    re1  = nn.ReLU()
    cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    bn2  = nn.BatchNorm2d(16)
    re2  = nn.ReLU()
    cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    
    mp1  = nn.MaxPool2d(kernel_size=2)
    cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,3), dilation=(4,4)) # no padding
    ## block 2
    bn3  = nn.BatchNorm2d(32)
    re3  = nn.ReLU()
    cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    bn4  = nn.BatchNorm2d(32)
    re4  = nn.ReLU()
    cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    
    mp2  = nn.MaxPool2d(kernel_size=2)
    cnn7 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,3), dilation=(8,8)) # no padding
    ## block 3
    bn5  = nn.BatchNorm2d(64)
    re5  = nn.ReLU()
    cnn8 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    bn6  = nn.BatchNorm2d(64)
    re6  = nn.ReLU()
    cnn9 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(2,2), dilation=(2,2))
    avg  = nn.AvgPool2d(kernel_size=3)
      
    # test 
    x = torch.Tensor(32,1,257,1091)
    x = cnn1(x)
    residual = x
    x = cnn3(re2(bn2(cnn2(re1(bn1(x))))))
    x += residual 
    print(x.size())
    x = cnn4(mp1(x))
    residual = x
    x = cnn6(re4(bn4(cnn5(re3(bn3(x))))))
    x += residual 
    print(x.size())
    x = cnn7(mp2(x))
    residual = x
    x = cnn9(re6(bn6(cnn8(re5(bn5(x))))))
    x += residual 
    print(x.size())
    x = avg(x)
    print(x.size())
    """
    
main()
