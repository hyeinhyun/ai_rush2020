#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:18:32 2020

@author: hihyun
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Densenet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.densenet201(pretrained=True)
        model = list(model.children())[:-1]
        self.net = nn.Sequential(*model)
        self.Linear1=nn.Linear(in_features=1920, out_features=out_size, bias=True)


    def forward(self, image):
        out = self.net(image)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0),-1)
        out = self.Linear1(out)
        return out

class Res50(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet50(pretrained=True)
        model = list(model.children())[:-1]
        self.net = nn.Sequential(*model)
        self.Linear1=nn.Linear(in_features=2048, out_features=out_size, bias=True)        

    def forward(self, image):
        out = self.net(image)
        out = out.view(out.size(0),-1)
        out = self.Linear1(out)
        return out

class wrn(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.wide_resnet50_2(pretrained=True)
        model = list(model.children())[:-1]
        self.net = nn.Sequential(*model)
        self.Linear1=nn.Linear(in_features=2048, out_features=out_size, bias=True)        

    def forward(self, image):
        out = self.net(image)
        out = out.view(out.size(0),-1)
        out = self.Linear1(out)
        return out
    
class Rex100(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnext101_32x8d(pretrained=True)
        model = list(model.children())[:-1]
        self.net = nn.Sequential(*model)
        self.Linear1=nn.Linear(in_features=2048, out_features=out_size, bias=True)        

    def forward(self, image):
        out = self.net(image)
        out = out.view(out.size(0),-1)
        out = self.Linear1(out)
        return out
    
class Res152(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet152(pretrained=True)
        model = list(model.children())[:-1]
        self.net = nn.Sequential(*model)
        self.Linear1=nn.Linear(in_features=2048, out_features=out_size, bias=True)        

    def forward(self, image):
        out = self.net(image)
        out = out.view(out.size(0),-1)
        out = self.Linear1(out)
        return out