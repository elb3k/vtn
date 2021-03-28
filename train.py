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
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils import load_yaml, GradualWarmupScheduler
from model import VTN

# Parse arguments
parser = ArgumentParser()

parser.add_argument("--annotations", type=str, default="dataset/kinetics-400/annotations.json", help="Dataset labels path")
parser.add_argument("--val-annotations", type=str, default="dataset/kinetics-400/annotations.json", help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="dataset/kinetics-400/train", help="Dataset files root-dir")
parser.add_argument("--val-root-dir", type=str, default="dataset/kinetics-400/val", help="Dataset files root-dir")
parser.add_argument("--classes", type=int, default=400, help="Number of classes")
parser.add_argument("--config", type=str, default='configs/vtn.yaml', help="Config file")

parser.add_argument("--dataset", choices=['ucf', 'smth', 'kinetics'], default='kinetics')
parser.add_argument("--weight-path", type=str, default="weights/kinetics/v2", help='Path to save weights')
parser.add_argument("--log-path", type=str, default="log/kinetics/v2", help='Path to save weights')
parser.add_argument("--resume", type=int, default=0, help='Resume training from')

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
parser.add_argument("--warmup_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--epochs", type=int, default=22, help="Number of epochs")
parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")

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
  train_set = Kinetics400(args.annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames)
  #train_set, val_set = random_split(dataset, [len(dataset) - int(len(dataset) * args.validation_split), int(len(dataset) * args.validation_split)], generator=torch.Generator().manual_seed(12345))
  val_set = Kinetics400(args.annotations, args.val_root_dir, preprocess=preprocess, frames=cfg.frames)

# Split
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=16, persistent_workers=True)

# Tensorboard 
tensorboard = SummaryWriter(args.log_path)

# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
softmax = nn.LogSoftmax(dim=1)

def adjust_learning_rate(optimizer, epoch, cur_iter, max_iter):

    """Sets the learning rate to the according to POLICY"""
    #for ind, step in enumerate(STEPS):
    #  if epoch < step:
    #    break
    #ind = ind - 1

    #lr = args.learning_rate * LRS[ind]
    
    # First 2 epochs warmup
    if epoch <= 2:
      # Linear warmup from warmup learning rate to learning rate
      cur_iter = (epoch-1) * max_iter + cur_iter
      lr = args.warmup_rate +  cur_iter / (max_iter * 2) * (args.learning_rate - args.warmup_rate) 
    
    else:
      # Cosine learning rate
      cur_iter = (epoch - 3) * max_iter + cur_iter
      full_iter = (args.epochs - 2) * max_iter
      
      # Minimum learning rate
      min_learning_rate = 1e-6

      lr = min_learning_rate + (args.learning_rate - min_learning_rate) * (np.cos(np.pi * cur_iter / full_iter) + 1.0) * 0.5
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
    


for epoch in range(max(args.resume+1, 1), args.epochs+1):
        
    # Adjust learning rate
    #scheduler = CosineAnnealingLR(optimizer, 100, 1e-4, -1)
    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch: {epoch}, loss: 0.000")
    for i, (src, target) in progress:
        
        lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader))
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

        # Summary
        if i % 100 == 99:
          tensorboard.add_scalar('train_loss', loss_val, epoch * len(train_loader) + i)
          tensorboard.add_scalar('lr', lr, epoch * len(train_loader) + i)

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
    # Summary
    tensorboard.add_scalar('val_loss', val_loss/len(val_loader), epoch)
    tensorboard.add_scalar('val_acc', val_acc/len(val_loader) * 100, epoch)

    # Save weights
    torch.save(model.state_dict(), f'{args.weight_path}/weights_{epoch}.pth')
