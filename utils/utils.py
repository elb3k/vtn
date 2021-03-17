import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

import yaml
from argparse import Namespace
import torchvision.transforms.functional as F

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_yaml(file):
    """
    Load yaml file
    """
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = Namespace(**cfg)
    return cfg



def getK(arr, k=16):

    out = []
    ratio = len(arr)/k

    for i in range(k):
        out.append(arr[int(i*ratio)])
    return out


def read_video(video, frames=16):

    imgs = []
    cap = cv2.VideoCapture(video)
    while True:
        status, img = cap.read()
        if not status:
            break
            
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return getK(imgs, frames)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)