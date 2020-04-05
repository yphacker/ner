# coding=utf-8
# author=yphacker

import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
from functools import partial
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_preds = F.log_softmax(output, dim=1)
        pt = torch.exp(log_preds)
        log_preds = (1 - pt) ** self.gamma * log_preds
        loss = F.nll_loss(log_preds, target, self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)
