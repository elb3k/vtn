import torch
from torch import nn, einsum
import torch.nn.functional as F
from argparse import Namespace
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from longformer import Longformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class VTN(nn.Module):
    def __init__(self, *, frames, num_classes, img_size, patch_size, spatial_args, temporal_args):
        super().__init__()
        self.frames = frames

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        #[Spatial] Transformer attention 
        self.spatial_transformer = timm.create_model(f'vit_small_patch{patch_size}_{img_size}', pretrained=True, **vars(spatial_args))
        
        # Spatial preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        self.preprocess = create_transform(**config)
        
        #Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)

        #[Temporal] Transformer_attention
        temporal_args.seq_len = frames
        self.temporal_transformer = Longformer(**vars(temporal_args))
        
        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, num_classes)
        )

    def forward(self, img):

        x = self.collapse_frames(img)
        
        # Spatial Transformer
        x = self.spatial_transformer.forward_features(x)

        # Spatial to temporal
        x = self.spatial2temporal(x)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # Classifier
        return self.mlp_head(x)
