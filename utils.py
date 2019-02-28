# -*- coding: UTF-8 -*-
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(y, prob_y, percentage=True):
    y = y.type(torch.float32)
    predicted_y = (torch.max(prob_y,1)[1]).type(torch.float32)
    acc = (y == predicted_y).sum(dtype=torch.float32) / float(y.size(0))
    if percentage:
        acc *= 100
    return acc