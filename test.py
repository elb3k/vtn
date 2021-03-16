import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import yaml
from argparse import ArgumentParser, Namespace
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from model import VTN
from utils.data import UCF101, SMTHV2
from torchvision import transforms

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_yaml

# Parse arguments
parser = ArgumentParser()

parser.add_argument("--annotations", type=str, default="dataset/smth/val.json", help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="dataset/smth/videos", help="Dataset files root-dir")
parser.add_argument("--classInd", type=str, default="dataset/ucf/annotation/classInd.txt", help="ClassInd file")
parser.add_argument("--classes", type=int, default=174, help="Number of classes")

parser.add_argument("--dataset", choices=['ucf', 'smth'], default='smth', help='Dataset type')
parser.add_argument("--weight-path", type=str, default="weights/smth/v1/weights_3.pth", help='Path to load weights')

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--config", type=str, default="configs/longVViT.yaml", help="Config file")



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

  dataset = UCF101(args.annotations, args.root_dir, preprocess=model.preprocess, classes=args.classes, frames=cfg.frames, train=False, class_map=class_map)

elif args.dataset == 'smth':
  dataset = SMTHV2(args.annotations, args.root_dir, preprocess=model.preprocess, frames=cfg.frames)

dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)

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
        src = src.cuda()
        target = target.cuda()
    
    with torch.no_grad():
        output = model(src)
        loss = loss_func(output, target)
        val_loss += loss.item()
        
        output = softmax(output)
        # Top 1
        top1_acc += torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item() / args.batch_size
        # Top 3
        _, idx = torch.topk(output, 5, dim=1)
        for label, top5 in zip(target, idx):
            if label in top5:
                top5_acc += 1/args.batch_size

val_loss = val_loss / len(dataloader)
top1_acc = top1_acc / len(dataloader)
top5_acc = top5_acc / len(dataloader)

print(f'Loss: {val_loss}, Top 1: {top1_acc}, Top 5: {top5_acc}')
