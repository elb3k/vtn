import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
import io
import cv2
import torchvision
import fs
import fs.copy
import av

from joblib import Parallel, delayed
from tqdm import tqdm
from functools import lru_cache

class UCF101(Dataset):


    def __init__(self, file, root_dir, preprocess=None, frames=16, classes=101, train=True, class_map=None, ):
        
        #torchvision.set_image_backend('accimage') 
        assert train or class_map is not None, "class_map is needed"

        with open(file, "r") as f:
            lines = [ tuple(line.strip().split()) for line in f.readlines()]  
        
        # Load with labels
        if train:
            self.src = [ (file, int(label)) for file, label in lines]
        else:
            self.src = [ (file[0], class_map[file[0].split("/")[0].strip()]) for file in lines]

        self.root_dir = root_dir
        self.classes = classes
        self.train = train
        self.frames = frames
        self.preprocess = preprocess

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.src[idx][0]
        
        imgs = []

        for i in range(self.frames):
            img = Image.open(f'{self.root_dir}/{filename}/{i}.jpg')

            if self.preprocess is not None:
                img = self.preprocess(img).unsqueeze(0)
            
            imgs += [img]
        
        if self.preprocess is not None:
            imgs = torch.cat(imgs)

        label = self.src[idx][1]
        
        return imgs, int(label-1)


def generate_seq(count, k):
  ratio = count / k 
  return list(map(lambda i: max(0, int(i*ratio)), range(0, k)))

def getK(arr, k=16):
  out = []
  ratio = len(arr)/k

  for i in range(k):
    out.append(arr[int(i*ratio)])
  return out

@lru_cache
def read_video(root, frames):
  
  cap = cv2.VideoCapture(root)
  imgs = []
  
  while True:
    status, img = cap.read()
    if not status:
      break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(Image.fromarray(img))
  
  assert len(imgs) > 0, f'{video} read error'
  return getK(imgs, frames)

class SMTHV2(Dataset):


    def __init__(self, labels, root_dir, preprocess=None, frames=16):


        with open(labels, "r") as f:
            src = json.load(f)
        
        self.src = [ (filename, label, tuple(generate_seq(count, frames))) for filename, label, count in src] 
        
        self.root_dir = root_dir
        self.frames = frames
        self.preprocess = preprocess

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read images
        id  = self.src[idx][0]
        label = self.src[idx][1]
        
        
        imgs = read_video(f'{self.root_dir}/{id}.webm', self.frames)
        
        if self.preprocess is not None:
            imgs = list(map(lambda img: self.preprocess(img).unsqueeze(0), imgs))
            imgs = torch.cat(imgs)
        
                
        return imgs, int(label)
