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
parser.add_argument("--classes", type=int, default=174, help="Number of classes")

parser.add_argument("--weight-path", type=str, default="weights/smth/v1/weights_3.pth", help='Path to load weights')
parser.add_argument("--output", type=str, default="submission.csv", help="Path to output file")

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--config", type=str, default="configs/vtn.yaml", help="Config file")



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

model.load_state_dict(torch.load(args.weight_path))
model.eval()


# Load dataset
dataset = SMTHV2(args.annotations, args.root_dir, preprocess=preprocess, frames=cfg.frames)

dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)

# Loss
loss_func = nn.CrossEntropyLoss()

# Softmax
softmax = nn.LogSoftmax(dim=1)

# Output
data = []

for src, target in tqdm(dataloader, desc=f"Generating {args.output}"):
    if torch.cuda.is_available():
        src = src.cuda()
        target = target.cuda()
    
    with torch.no_grad():
        output = model(src)
        loss = loss_func(output, target)
        val_loss += loss.item()
        
        output = softmax(output).cpu().detach()
        # Top 5
        _, idx = torch.topk(output, 5, dim=1)
        # Add to output
        for id, top5 in zip(labels[i*args.batch_size: (i+1)*args.batch_size], idx):
            data += [ [id] + top5.numpy().tolist()]

# Write to output
df = pd.DataFrame(data)
df.to_csv(args.output, header=False, index=False, sep=';')
