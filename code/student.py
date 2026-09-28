"""Student GPT with SwiGLU, LayerNorm, and configurable positions.

The model retains causal attention, residual structure, weight tying, and
the supplied model interface.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class SwiGLU(nn.Module):
    """Parameter-matched gated feed-forward network."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))


class RotaryEmbedding(nn.Module):
    """Precomputed interleaved rotary position embeddings for one head."""

    def __init__(self, head_dim, context, theta=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError('RoPE requires an even attention head dimension.')
        if theta <= 0.0:
            raise ValueError('rope_theta must be positive.')
        positions = torch.arange(context, dtype=torch.float32)
        frequencies = theta ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        angles = torch.outer(positions, frequencies)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):
        length = x.shape[-2]
        if length > self.cos.shape[0]:
            raise ValueError('Input sequence exceeds the configured context.')
        cos = self.cos[:length].to(dtype=x.dtype)[None, None, :, :]
        sin = self.sin[:length].to(dtype=x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)


class SwiGLUBlock(nn.Module):
    """SwiGLU block that reuses baseline attention projections."""

    def __init__(self, baseline_block, width, hidden,
                 position_encoding, context, rope_theta, qk_norm, dropout):
        super().__init__()
        self.heads = baseline_block.heads
        if width % self.heads != 0:
            raise ValueError('Model width must be divisible by the number of heads.')
        self.norm1 = baseline_block.norm1
        self.norm2 = baseline_block.norm2
        self.qkv = baseline_block.qkv
        self.proj = baseline_block.proj
        self.mlp = SwiGLU(width, hidden)
        for layer in self.mlp.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=.02)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.rope = (
            RotaryEmbedding(width // self.heads, context, rope_theta)
            if position_encoding == 'rope' else None
        )
        head_dim = width // self.heads
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else None

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention_update = self.proj(
            attended.transpose(1, 2).reshape(batch, length, width)
        )
        x = x + self.attn_dropout(attention_update)
        x = x + self.ffn_dropout(self.mlp(self.norm2(x)))
        return x


class StudentGPT(GPT):
    """The supplied GPT with configurable modernized sublayers."""

    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        hidden = config.get('ffn_hidden')
        if hidden is None:
            hidden = int(math.ceil((8 * width / 3) / 8) * 8)
        if config.get('norm', 'layernorm') != 'layernorm':
            raise ValueError('Only layernorm is supported by this implementation.')
        dropout = float(config.get('resid_dropout', 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError('resid_dropout must be in [0, 1).')
        position_encoding = config.get('position_encoding', 'learned')
        rope_theta = float(config.get('rope_theta', 10000.0))
        qk_norm = bool(config.get('qk_norm', False))
        if position_encoding not in ('learned', 'rope'):
            raise ValueError('position_encoding must be learned or rope.')
        baseline_blocks = list(self.blocks)
        self.blocks = nn.ModuleList([
            SwiGLUBlock(
                block, width, hidden,
                position_encoding, config['context'], rope_theta, qk_norm,
                dropout
            )
            for block in baseline_blocks
        ])
        self.position_encoding = position_encoding
        if position_encoding == 'rope':
            self.pos = None

    def features(self, ids):
        x = self.token(ids)
        if self.position_encoding == 'learned':
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config):
    return StudentGPT(config)
