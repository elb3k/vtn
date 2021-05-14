import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

import yaml
from argparse import ArgumentParser, Namespace
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from model import VTN
from utils.data import UCF101, SMTHV2, Kinetics400
from utils.utils import preprocess
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_yaml
from einops import rearrange
import json
from glob import glob

# Parse arguments
parser = ArgumentParser()

parser.add_argument("--annotations", type=str, default="dataset/kinetics-400/annotations.json", help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="dataset/kinetics-400/val", help="Dataset files root-dir")
parser.add_argument("--classInd", type=str, default="dataset/ucf/annotation/classInd.txt", help="ClassInd file")
parser.add_argument("--classes", type=int, default=400, help="Number of classes")
parser.add_argument("--dataset", choices=['ucf', 'smth', 'kinetics'], default='kinetics', help='Dataset type')
parser.add_argument("--per_sample", type=int, default=2, help="Clips per sample")
parser.add_argument("--weight-path", type=str, default="weights/kinetics/lin-v3/weights_20.pth", help='Path to load weights')
# Hyperparameters
parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
parser.add_argument("--config", type=str, default="configs/lin-vtn.yaml", help="Config file")



# Parse arguments
args = parser.parse_args()
print(args)

# Load config
cfg = load_yaml(args.config)

# Load model
model = VTN(**vars(cfg))

if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()

model.load_state_dict(torch.load(args.weight_path))
model.eval()


# Load dataset
if args.dataset == 'ucf':
  # Load class name to index
  class_map = {}
  with open(args.classInd, "r") as f:
    for line in f.readlines():
        index, name = line.strip().split()
        index = int(index)
        class_map[name] = index

  dataset = UCF101(args.annotations, args.root_dir, preprocess=preprocess, classes=args.classes, frames=cfg.frames, train=False, class_map=class_map)

elif args.dataset == 'smth':
  dataset = SMTHV2(args.annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames)

elif args.dataset == 'kinetics':

  import av
  from functools import lru_cache
  from torchvision import transforms

  def getK(arr, k=16):
    out = []
    ratio = len(arr)/k

    for i in range(k):
      out.append(arr[int(i*ratio)])
    return out

  @lru_cache
  def read_video(root, frames, target=5.12, per_sample=10):
    
    try:
      # Read video
      cap = av.open(root)

      # Metadata
      fps = float(cap.streams.video[0].average_rate)
      duration = cap.streams.video[0].frames / fps
      target_fps = frames/target

      # Number of new frames
      new_frames = int(target_fps * duration) if duration>=target else frames
      imgs = getK([ img.to_image() for img in cap.decode(video=0)], k=new_frames)
      diff = (new_frames - frames) / max(per_sample - 1, 1)

      # Generate imgs
      out = []
      for i in range(per_sample):
        start = int(i*diff)
        out.append(imgs[start: start+frames])
      
      return out
      


    except Exception as e:
      print(f"Read error of video {root}, {e}")

  # Kinetics-400
  class Kinetics400(Dataset):
    
    def __init__(self, labels, root_dir, preprocess=None, frames=16, per_sample=1):
      
      assert per_sample > 0
      with open(labels, "r") as f:
        labels = json.load(f)
      
      files = glob(f"{root_dir}/*/*")
      self.src = [ (file, labels[file.split('/')[-2]] )  for file in files ]
      self.frames = frames
      self.preprocess = preprocess
      self.per_sample = per_sample

      self.resize = transforms.Resize(256)
      self.three_crop = transforms.Compose([
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: [ crop for i, crop in enumerate(crops) if i in [0, 3, 4] ])
      ])

      self.preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
      ])

    def __len__(self):
      return len(self.src)

    def __getitem__(self, idx):
      
      if torch.is_tensor(idx):
        idx = idx.tolist()
      
      id, label = self.src[idx]
      
      videos = read_video(id, self.frames, per_sample=self.per_sample)

      if self.preprocess is not None:
        out = []
        for imgs in videos:
          three_imgs = list(map(lambda img: self.three_crop(self.resize(img)), imgs))

          for j in range(3):
            imgs = [ self.preprocess(three_imgs[i][j]).unsqueeze(0)  for i in range(len(three_imgs))]  
            imgs = torch.cat(imgs)
            out.append(imgs.unsqueeze(0))
      
      return torch.cat(out), int(label)
  dataset = Kinetics400(args.annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames, per_sample=args.per_sample)

dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

# Loss
loss_func = nn.CrossEntropyLoss()

# Softmax
softmax = nn.LogSoftmax(dim=1)

# Validation
val_loss = 0
top1_acc = 0
top5_acc = 0


for src, target in tqdm(dataloader, desc="Validating"):
    # src, target = train_loader[i]
    if torch.cuda.is_available():
        # print(src.shape)
        src = rearrange(src, 'b p f c h w -> (b p) f c h w')
        src = src.cuda()
        target = target.cuda()
    
    with torch.no_grad():
        output = model(src)
        # Rearrange
        output = torch.mean(rearrange(output, '(b p) d -> b p d', p=args.per_sample*3), dim=1)

        loss = loss_func(output, target)
        val_loss += loss.item()

        output = softmax(output)
        # Top 1
        top1_acc += torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item()
        # Top 5
        _, idx = torch.topk(output, 5, dim=1)
        for label, top5 in zip(target, idx):
          if label in top5:
            top5_acc += 1
        

count = len(dataloader) * args.batch_size

val_loss = val_loss / len(dataloader)
top1_acc = top1_acc / count
top5_acc = top5_acc / count

print(f'Loss: {val_loss}, Top 1: {top1_acc}, Top 5: {top5_acc}')
