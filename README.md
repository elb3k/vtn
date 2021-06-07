# VTN - Pytorch
Implemenetation of [Video Transformer Network](https://arxiv.org/abs/2102.00719), a simple framework for video classification task, with [Vision Transformer](https://arxiv.org/abs/2010.11929) backbone, with additional temporal transformers.

### Spatial Backbone:
Visual Transformer - using [timm](https://github.com/rwightman/pytorch-image-models), can be changed to any image classifier


### Temporal Backbone:
1. Longformer - original transformer used in a paper, [sample config](configs/vtn.yaml)
2. Linformer - another linear complexity transformer for my own research, [sample config](configs/lin-vtn.yaml)
3. Transformer - simple full transformer encoder, with a right configuration, model can be used as implementation of [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095), [sample config](configs/full-vtn.yaml)

### Dataset implemenatations:
Basic dataset loaders for
1. Kinetics-400, (can be used for any `Kinetics-xxx` dataset)
2. Something-Something-V2
3. UCF-101

### Usage
```python
import torch
from utils import load_yaml
from model import VTN

cfg = load_yaml('configs/vtn.yaml')

model = VTN(**vars(cfg))

video = torch.rand(1, 16, 3, 224, 224)

preds = model(video) # (1, 400)
```

Parameters are self-explanatory in config file


### Results
| Model             | Top-1 | Top-5 | Weights |
| ----------        | ----- | ----- | ------- |
| Longformer-VTN    | 78.9% | 93.7% | taken [from](https://arxiv.org/abs/2102.00719) |
| Transformer-VTN   | 78.0% | 93.7% | taken [from](https://arxiv.org/abs/2102.05095) |
| Linformer-VTN     | 75.6% | 92.6% | [link](https://github.com/elb3k/vtn/releases/download/weights/lin-vtn.pth) |
| Linformer-VTN-MIIL-21k | 76.8% | 93.4% | [link](https://github.com/elb3k/vtn/releases/download/weights/lin-vtn-miil_21k.pth) |
| Linformer-VTN-21k | 77.8% | 93.4% | |
