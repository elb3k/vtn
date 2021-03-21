import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from argparse import ArgumentParser
import torch
torch.backends.cudnn.benchmark = True
import yaml

import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from utils import UCF101, SMTHV2, Kinetics400
from torchvision import transforms

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils import load_yaml, GradualWarmupScheduler
from model import VTN

# Parse arguments
parser = ArgumentParser()

parser.add_argument("--annotations", type=str, default="dataset/kinetics-400/annotations.json", help="Dataset labels path")
parser.add_argument("--val-annotations", type=str, default="dataset/kinetics-400/annotations.json", help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="dataset/kinetics-400/train", help="Dataset files root-dir")
parser.add_argument("--classes", type=int, default=400, help="Number of classes")
parser.add_argument("--config", type=str, default='configs/vtn.yaml', help="Config file")

parser.add_argument("--dataset", choices=['ucf', 'smth', 'kinetics'], default='kinetics')
parser.add_argument("--weight-path", type=str, default="weights/kinetics/v1", help='Path to save weights')
parser.add_argument("--resume", type=int, default=0, help='Resume training from')

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--epochs", type=int, default=22, help="Number of epochs")
parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split")

# Learning scheduler
LRS = [1, 0.1, 0.01]
STEPS = [1, 7, 15]

# Parse arguments
args = parser.parse_args()
print(args)

# Load config
cfg = load_yaml(args.config)

# Load model
model = VTN(**vars(cfg))
preprocess = model.preprocess

if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()

# Resume weights
if args.resume > 0:
  model.load_state_dict(torch.load(f'{args.weight_path}/weights_{args.resume}.pth'))   

# Load dataset
if args.dataset == 'ucf':
  train_set = UCF101(args.annotations, args.root_dir, preprocess=preprocess, classes=args.classes, frames=cfg.frames)
  val_set = UCF101(args.val_annotations, args.root_dir, preprocess=preprocess, classes=args.classes, frames=cfg.frames) 
elif args.dataset == 'smth':
  train_set = SMTHV2(args.annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames)
  val_set = SMTHV2(args.val_annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames)
elif args.dataset == 'kinetics':
  dataset = Kinetics400(args.annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames)
  train_set, val_set = random_split(dataset, [len(dataset) - int(len(dataset) * args.validation_split), int(len(dataset) * args.validation_split)], generator=torch.Generator().manual_seed(12345))

# Split
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=8, persistent_workers=True)


# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
softmax = nn.LogSoftmax(dim=1)

def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the according to POLICY"""
    for ind, step in enumerate(STEPS):
      if epoch < step:
        break
    ind = ind - 1

    lr = args.learning_rate * LRS[ind]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



for epoch in range(max(args.resume+1, 1), args.epochs+1):
        
    # Adjust learning rate
    #scheduler = CosineAnnealingLR(optimizer, 100, 1e-4, -1)
    adjust_learning_rate(optimizer, epoch)
    progress = tqdm(train_loader, desc=f"Epoch: {epoch}, loss: 0.000")
    for src, target in progress:
        
        # print(src.shape, target.shape)
        # src, target = train_loader[i] 
        if torch.cuda.is_available():
            src = torch.autograd.Variable(src).cuda()
            target = torch.autograd.Variable(target).cuda()
        optimizer.zero_grad()
        # Forward + backprop + optimize
        output = model(src)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        # Cosine scheduler
        #scheduler.step()
 
        # Show loss
        loss_val = loss.item()
        progress.set_description(f"Epoch: {epoch}, loss: {loss_val}")

    # Validation
    val_loss = 0
    val_acc = 0
    for src, target in tqdm(val_loader, desc=f"Epoch: {epoch}, validating"):
        # src, target = train_loader[i]
        if torch.cuda.is_available():
            src = torch.autograd.Variable(src).cuda()
            target = torch.autograd.Variable(target).cuda()
        
        with torch.no_grad():
            output = model(src)
            loss = loss_func(output, target)
            val_loss += loss.item()
            output = softmax(output)
            
            val_acc += torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item() / args.batch_size

    print("Validating loss:", val_loss/len(val_loader), ", accuracy:", val_acc/len(val_loader))

    # Save weights
    torch.save(model.state_dict(), f'{args.weight_path}/weights_{epoch}.pth')
