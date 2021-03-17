import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

import yaml
from argparse import Namespace
import torchvision.transforms.functional as F


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
