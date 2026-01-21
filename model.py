"""
Masked Diffusion Transformer - Bidirectional transformer for masked diffusion LM.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head, self.head_dim = config.n_head, config.head_dim
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # Bidirectional!
        return self.dropout(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = ((int(8 / 3 * config.n_embd) + 255) // 256) * 256
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w3 = nn.Linear(hidden, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.n_embd)  # +1 for mask
        self.drop = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(config.head_dim, config.block_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('w3.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"Model: {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, mask_indices=None, p_mask=None):
        B, T = x.shape
        h = self.drop(self.tok_emb(x))
        cos, sin = self.rotary(h, T)
        for block in self.blocks:
            h = block(h, cos, sin)
        logits = self.head(self.norm(h))

        loss = None
        if targets is not None and mask_indices is not None:
            loss_unreduced = F.cross_entropy(
                logits.view(-1, self.config.vocab_size)[mask_indices.view(-1)],
                targets.view(-1)[mask_indices.view(-1)],
                reduction='none'
            )
            if p_mask is not None:
                loss_unreduced = loss_unreduced / p_mask.view(-1)[mask_indices.view(-1)]
            loss = loss_unreduced.sum() / (B * T)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2 and p.requires_grad]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2 and p.requires_grad]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        fused = device_type == "cuda" and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)
