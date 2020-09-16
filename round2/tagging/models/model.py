import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn


class Resnet50_FC2(torch.nn.Module):
    def __init__(self, n_class=9, pretrained=True):
        super(Resnet50_FC2, self).__init__()
        self.basemodel = models.resnet50(pretrained=pretrained)
        self.linear1 = torch.nn.Linear(1000, 512)
        self.linear2 = torch.nn.Linear(512, n_class)

    def forward(self, x):
        x = self.basemodel(x)
        x = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(x), dim=-1)
        pred = torch.argmax(out, dim=-1)
        return out, pred

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

class InceptionV3(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.net = models.inception_v3(pretrained=True)
        self.Linear1=nn.Linear(in_features=1000, out_features=out_size, bias=True)

    def forward(self, image):
        h,out = self.net(image)
        out = out.view(out.size(0),-1)
        out = self.Linear1(out)
        return out
