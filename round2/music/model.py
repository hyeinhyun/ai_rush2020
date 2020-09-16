import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch
from torch import nn
import torch.nn.functional as F

num_classes=4



# Source: https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Source: https://github.com/luuuyi/CBAM.PyTorch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SkipBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AuxBlock(nn.Module):
    def __init__(self, last_fc, num_classes, base_size, dropout):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*8, base_size*last_fc),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*last_fc),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*last_fc, num_classes),
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AuxSkipAttention(nn.Module):
    def __init__(self, num_classes, base_size=64,
                 dropout=0.2, ratio=16, kernel_size=7,
                 last_filters=8, last_fc=2):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=base_size)
        self.skip1 = SkipBlock(in_channels=base_size, out_channels=base_size*8,
                               scale_factor=8)

        self.conv2 = ConvBlock(in_channels=base_size, out_channels=base_size*2)
        self.skip2 = SkipBlock(in_channels=base_size * 2, out_channels=base_size*8,
                               scale_factor=4)

        self.conv3 = ConvBlock(in_channels=base_size*2, out_channels=base_size*4)
        self.skip3 = SkipBlock(in_channels=base_size*4, out_channels=base_size*8,
                               scale_factor=2)

        self.conv4 = ConvBlock(in_channels=base_size*4, out_channels=base_size*8)

        self.attention = ConvolutionalBlockAttentionModule(base_size*8*4,
                                                           ratio=ratio,
                                                           kernel_size=kernel_size)
        self.merge = SkipBlock(base_size*8*4, base_size*last_filters, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size*last_filters, base_size*last_fc),
            nn.PReLU(),
            nn.BatchNorm1d(base_size*last_fc),
            nn.Dropout(dropout/2),
            nn.Linear(base_size*last_fc, num_classes),
        )

        self.aux1 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux2 = AuxBlock(last_fc, num_classes, base_size, dropout)
        self.aux3 = AuxBlock(last_fc, num_classes, base_size, dropout)

    def forward(self, x):
        x = self.conv1(x)
        skip1 = self.skip1(x)
        aux1 = self.aux1(skip1)

        x = self.conv2(x)
        skip2 = self.skip2(x)
        aux2 = self.aux2(skip2)

        x = self.conv3(x)
        skip3 = self.skip3(x)
        aux3 = self.aux3(skip3)

        x = self.conv4(x)

        x = torch.cat([x, skip1, skip2, skip3], dim=1)

        x = self.attention(x)
        x = self.merge(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self._conv = nn.Sequential(
            # conv1_block
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),

            # conv2_block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), (4, 4)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 4), (4, 4)),
        )
        self._clf1 = nn.GRU(512, 256)
        self._clf2 = nn.GRU(256, 128,dropout=0.3)
        self._lin = nn.Sequential(nn.Linear(128, 4))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # (batch,1,512,1200(time))
        x = self._conv(x)
        x = x.permute((0,3, 2, 1))  # (batch,time(62),512)
        x = x.reshape((-1,62, 512))
        out,hidden = self._clf1(x)
        out,hidden = self._clf2(out)
        out = self._lin(out[:,-1,:])
        return out

class CRNN2(nn.Module):
    def __init__(self):
        super().__init__()

        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((4, 8)),
            nn.Dropout(0.1),
        )
        self._clf1 = nn.GRU(1024, 512)
        self._clf2 = nn.GRU(512, 128,dropout=0.3)
        self._lin = nn.Sequential(nn.Linear(128, 4))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # (batch,1,128,1200(time))
        x = self._conv(x)
        x = x.permute((0,3, 2, 1))  # (batch,time(32),1024)
        #print(x.shape)
        x = x.reshape((-1,1, 1024))
        #print(x.shape)
        out,hidden = self._clf1(x)
        out,hidden = self._clf2(out)
        out = self._lin(out[:,-1,:])
        return out


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

        )
        self.lin1 = nn.Sequential(nn.Linear(76800, 4096),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),

                                  )
        self.lin2 = nn.Sequential(nn.Linear(4096, 5))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv(x)
        x = x.reshape(-1, 76800)
        x = self.lin1(x)
        x = self.lin2(x)

        return x
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        #128,1200
        self._clf1 = nn.LSTM(128, 256,3)
        self._lin = nn.Sequential(nn.Linear(256, 4))

    def forward(self, x):
        # (batch,128,256)
        #x = x.permute((0,2, 1))# (batch,256,128)
        x= torch.transpose(x,1,2)
        hidden = Variable(torch.zeros(3,256,256)).cuda() # (num_layers * num_directions, batch, hidden_size)
        cell = Variable(torch.zeros(3,256,256)).cuda() # (num_layers * num_directions, batch, hidden_size)        out,hidden = self._clf1(x,h0)
        out,hidden = self._clf1(x,(hidden,cell))#batch*7*3
        out = self._lin(out[:,-1,:])
        return out


