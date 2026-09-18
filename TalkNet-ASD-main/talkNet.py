import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel
class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()

        self.model = talkNetModel()
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr = lr
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size = 1,
            gamma = lrDecay
        )

        print(
            time.strftime("%m-%d %H:%M:%S")
            + " Model para number = %.2f"
            % (
                sum(
                    param.numel()
                    for param in self.model.parameters()
                ) / 1024 / 1024
            )
        )