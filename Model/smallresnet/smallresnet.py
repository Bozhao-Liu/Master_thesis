# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

def Same_padding_conv2d(in_channels, out_channels, kernel_size=3):
    padding = int((kernel_size-1)/2)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                     padding=padding, groups=1, bias=True, dilation=1)
    
def down2_conv2d(in_channels, out_channels, kernel_size = 5):
    stride = int((kernel_size-1)/2)
    padding = int(stride)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=1, bias=True, dilation=1)

class IdentityBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(IdentityBlock, self).__init__()
        mid_channel = round((in_channels + out_channels)/2)
        self.Convpass = nn.Sequential(
            Same_padding_conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size = kernel_size),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            Same_padding_conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size = kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            down2_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size = kernel_size),
        )
        self.identityPass = down2_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size)
        
    def forward(self, x):
        identity = self.identityPass(x)

        out = self.Convpass(x)

        out += identity

        return out

class ResNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),  #256*256 -> 128*128
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                                           #128*128 -> 64*64
            IdentityBlock(in_channels=64, out_channels=256, kernel_size=5),                             #64*64 -> 32*32
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                                           #32*32 -> 16*16
            IdentityBlock(in_channels=256, out_channels=512, kernel_size=7),                            #16*16 -> 8*8
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),                                                      #8*8 -> 4*4
            nn.Conv2d(512, 512, kernel_size=3, stride=1),                                               #4*4 -> 2*2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                                                                #2*2 -> 1*1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x
    
class resize(nn.Module):
    def __init__(self):
        super(resize, self).__init__()

    def forward(self, x):
        x = nn.Upsample((256,256))(x)
        return x  

def smallresnet(num_classes = 1, **kwargs):
    return nn.Sequential(resize(), ResNet(num_classes, **kwargs))
