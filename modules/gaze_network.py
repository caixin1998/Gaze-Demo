
import torch
import torch.nn as nn
from torch.nn import init
import functools
from .resnet import resnet50, resnet18


class GazeNetwork(nn.Module):
    def __init__(self, opt):
        super(GazeNetwork, self).__init__()
        self.opt = opt
        if opt.backbone == "resnet50":
            self.model = resnet50(pretrained=True)
        elif opt.backbone == "resnet18":
            self.model = resnet18(pretrained=True)
        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, opt.ngf),
            nn.ReLU(inplace = True),
            nn.Linear(opt.ngf, 2),
        )

    def forward(self, x): 
        x1 = self.model(x["face"])
        x2 = self.gaze_fc(x1)
        return {"gaze":x2}