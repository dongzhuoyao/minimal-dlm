"""
dKV-Cache: Delayed Key-Value Cache for Diffusion Language Models.

Adapted from: https://github.com/horseee/dKV-Cache
Paper: dKV-Cache: The Cache for Diffusion Language Models (NeurIPS'25)

Key insight: Token representations stabilize after being decoded (unmasked),
so we cache K,V only for decoded tokens and update as the mask pattern changes.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch


class DKVCache:
    """
    Dynamic KV-Cache for masked diffusion models.

    Unlike autoregressive KV-cache which grows sequentially, this cache
    tracks decoded (unmasked) tokens and handles the dynamic reordering
    as mask patterns change during the diffusion reverse process.
    """

    def __init__(self, num_layers: int, device: torch.device = None):
        self.num_layers = num_layers
        self.device = device

        # Per-layer K,V caches for decoded tokens
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

        # Transfer order for reordering cache when mask pattern changes
        self._transfer_order: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        """Check if cache has been populated."""
        return self.key_cache[0] is None

    def reset(self) -> None:
        """Clear all cached values."""
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers
        self._transfer_order = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the cached sequence length for a layer."""
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def _compute_transfer_order(
        self,
        cache_position: torch.Tensor,
        prv_cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reordering indices when transitioning from previous to current mask state.

        Args:
            cache_position: Current decoded positions (B, T) bool tensor, True = decoded
            prv_cache_position: Previous decoded positions (B, T) bool tensor

        Returns:
            Transfer order indices for reordering (B, T)
        """
        B = cache_position.shape[0]

        # Current ordering: [masked positions, decoded positions]
        current_order = torch.cat([
            (~prv_cache_position).nonzero(as_tuple=True)[1].view(B, -1),
            prv_cache_position.nonzero(as_tuple=True)[1].view(B, -1),
        ], dim=-1)

        # Next ordering: [masked positions, decoded positions]
        next_order = torch.cat([
            (~cache_position).nonzero(as_tuple=True)[1].view(B, -1),
            cache_position.nonzero(as_tuple=True)[1].view(B, -1),
        ], dim=-1)

        # Compute transfer order
        transfer_order = []
        for b in range(B):
            value_to_index = {v.item(): i for i, v in enumerate(current_order[b])}
            indices = torch.tensor(
                [value_to_index.get(v.item(), -1) for v in next_order[b]],
                device=cache_position.device
            )
            transfer_order.append(indices)

        return torch.stack(transfer_order, dim=0)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
        prv_cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            key_states: New key states (B, n_heads, T, head_dim)
            value_states: New value states (B, n_heads, T, head_dim)
            layer_idx: Which transformer layer
            cache_position: Current decoded positions (B, T) bool tensor, True = decoded
            prv_cache_position: Previous decoded positions, None for first step

        Returns:
            Updated key and value states to use for attention
        """
        B, n_h, T, h_d = key_states.shape

        if prv_cache_position is None:
            # First step: cache decoded positions directly
            if self.is_empty():
                # Select and cache decoded token K,V
                key_selected = key_states[
                    cache_position[:, None, :, None].expand_as(key_states)
                ].view(B, n_h, -1, h_d)
                value_selected = value_states[
                    cache_position[:, None, :, None].expand_as(value_states)
                ].view(B, n_h, -1, h_d)

                self.key_cache[layer_idx] = key_selected
                self.value_cache[layer_idx] = value_selected
        else:
            # Subsequent steps: reorder based on transfer order
            if layer_idx == 0:
                # Compute transfer order once per step (reused across layers)
                self._transfer_order = self._compute_transfer_order(
                    cache_position, prv_cache_position
                )

            transfer_order = self._transfer_order

            # Reorder key/value states according to transfer order
            key_states = torch.gather(
                key_states, 2,
                transfer_order[:, None, :, None].expand_as(key_states)
            )
            value_states = torch.gather(
                value_states, 2,
                transfer_order[:, None, :, None].expand_as(value_states)
            )

            # Update cache with decoded portion
            cache_start = torch.sum(~cache_position, dim=-1)[0]
            self.key_cache[layer_idx] = key_states[:, :, cache_start:, :]
            self.value_cache[layer_idx] = value_states[:, :, cache_start:, :]

            # Clear transfer order after last layer
            if layer_idx == self.num_layers - 1:
                self._transfer_order = None

        return key_states, value_states


class RotaryEmbeddingCache:
    """
    Cache for rotary position embeddings that handles dynamic mask patterns.

    When using dKV-cache, we need position embeddings that match the
    reordered sequence (masked tokens first, decoded tokens second).
    """

    def __init__(self):
        self.cos_cache: Optional[torch.Tensor] = None
        self.sin_cache: Optional[torch.Tensor] = None
        self.pos_cos_masked: Optional[torch.Tensor] = None
        self.pos_sin_masked: Optional[torch.Tensor] = None
        self.pos_cos_decoded: Optional[torch.Tensor] = None
        self.pos_sin_decoded: Optional[torch.Tensor] = None

    def set_base_cache(self, cos: torch.Tensor, sin: torch.Tensor):
        """Set the base rotary embeddings."""
        # Expand to (1, 1, T, head_dim) for broadcasting
        self.cos_cache = cos.unsqueeze(0).unsqueeze(0) if cos.dim() == 2 else cos
        self.sin_cache = sin.unsqueeze(0).unsqueeze(0) if sin.dim() == 2 else sin

    def compute_reordered_positions(
        self,
        decoded_mask: torch.Tensor,
        n_heads: int,
    ):
        """
        Compute position embeddings reordered for cache usage.

        Args:
            decoded_mask: Bool tensor (B, T) where True = decoded token
            n_heads: Number of attention heads
        """
        if self.cos_cache is None:
            raise RuntimeError("Base cache not set. Call set_base_cache first.")

        B, T = decoded_mask.shape
        _, _, _, h_d = self.cos_cache.shape

        # Get reordering indices
        decoded_mask_expanded = decoded_mask.view(B, 1, T, 1).expand(B, n_heads, T, h_d)

        # Expand base cache for batch
        cos = self.cos_cache[:, :, :T, :].expand(B, n_heads, T, h_d)
        sin = self.sin_cache[:, :, :T, :].expand(B, n_heads, T, h_d)

        # Extract positions for masked and decoded tokens
        self.pos_cos_masked = cos[~decoded_mask_expanded].view(B, n_heads, -1, h_d)
        self.pos_sin_masked = sin[~decoded_mask_expanded].view(B, n_heads, -1, h_d)
        self.pos_cos_decoded = cos[decoded_mask_expanded].view(B, n_heads, -1, h_d)
        self.pos_sin_decoded = sin[decoded_mask_expanded].view(B, n_heads, -1, h_d)

    def get_q_pos_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get position embeddings for Q (masked tokens only)."""
        return self.pos_cos_masked, self.pos_sin_masked

    def get_k_pos_emb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get position embeddings for K (masked + decoded)."""
        cos = torch.cat([self.pos_cos_masked, self.pos_cos_decoded], dim=2)
        sin = torch.cat([self.pos_sin_masked, self.pos_sin_decoded], dim=2)
        return cos, sin

    def reset(self):
        """Clear cached positions."""
        self.pos_cos_masked = None
        self.pos_sin_masked = None
        self.pos_cos_decoded = None
        self.pos_sin_decoded = None
