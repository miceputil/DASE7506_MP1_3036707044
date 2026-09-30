"""Shared implementation entry point for final-configuration ablations."""
import copy
import torch
from torch import nn

from student import StudentGPT


def build_model(config):
    config = copy.deepcopy(config)
    variant = config.pop('ablation', 'full')

    if variant == 'without_rope':
        config['position_encoding'] = 'learned'
    elif variant == 'without_qk_norm':
        config['qk_norm'] = False
    elif variant == 'without_dropout':
        config['resid_dropout'] = 0.0
    elif variant not in ('full', 'without_swiglu'):
        raise ValueError(f'Unknown ablation variant: {variant}')

    model = StudentGPT(config)
    if variant == 'without_swiglu':
        width = config['width']
        hidden = 4 * width
        for block in model.blocks:
            mlp = nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Linear(hidden, width),
            )
            for layer in mlp.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)
            block.mlp = mlp
    return model
