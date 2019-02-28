import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv_layer1 = nn.Sequential( # input shape(1,28.28)
            nn.Conv2d(in_channels=1,  # input 是单通道
                      out_channels=16, # n_filters
                      kernel_size=5, # filter size
                      stride=1, #filter step
                      padding=2,
                      ),               # outputshape(16,28,28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)) # outputshape(16,14,14)

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,  # (16,14,14)
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2), # shape(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)) # shape(32,7,7)
        # output 10 classes(0~9)
        self.out = nn.Linear(in_features=32*7*7,out_features=10)

    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x) # (batch,32,7,7)
        x = x.view(x.size(0),-1) # (batch,32*7*7)
        output = self.out(x)
        return output,x

    



