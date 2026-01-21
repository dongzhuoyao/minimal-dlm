"""
Masked Diffusion Process - Forward masking and reverse unmasking for discrete tokens.

Supports dKV-Cache for accelerated inference (NeurIPS'25).
Reference: https://github.com/horseee/dKV-Cache
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MaskedDiffusion:
    """
    Forward: q(x_t | x_0) = Mask tokens with p_t = (1-eps)*t + eps
    Loss: L = E[CE(model(x_t), x_0) / p_t]  (importance weighting)
    """
    def __init__(self, vocab_size: int, eps: float = 1e-3):
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size
        self.eps = eps

    def forward_process(self, x_0: torch.Tensor, t: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply forward diffusion (masking). Returns (x_t, mask_indices, p_mask)."""
        B, T = x_0.shape
        if t is None:
            t = torch.rand(B, device=x_0.device)
        p_mask = ((1 - self.eps) * t + self.eps).unsqueeze(1).expand(B, T)
        mask_indices = torch.rand(B, T, device=x_0.device) < p_mask
        x_t = torch.where(mask_indices, self.mask_token_id, x_0)
        return x_t, mask_indices, p_mask

    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Precompute number of tokens to transfer at each step for uniform scheduling.

        From dKV-Cache: Because MDM employs a linear noise schedule,
        the expected number of tokens transitioned at each step should be consistent.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        ) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens

    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply Gumbel noise for sampling from categorical distribution.

        From dKV-Cache paper: For MDM, using float64 for Gumbel max
        improves generation quality.
        """
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @torch.no_grad()
    def sample(
        self,
        model,
        batch_size: int,
        seq_len: int,
        steps: int = 64,
        temperature: float = 1.0,
        device: str = "cuda",
        prompt: Optional[torch.Tensor] = None,
        strategy: str = "stochastic",
        use_dkv_cache: bool = False,
        cache_reloading_step: int = 1,
    ) -> torch.Tensor:
        """
        Generate by iterative unmasking with optional dKV-Cache acceleration.

        Args:
            model: The MaskedDiffusionTransformer model
            batch_size: Number of sequences to generate
            seq_len: Length of sequences to generate
            steps: Number of diffusion steps
            temperature: Sampling temperature (0 = greedy)
            device: Device to generate on
            prompt: Optional prompt tensor (B, prompt_len)
            strategy: 'stochastic' or 'confidence' unmasking
            use_dkv_cache: Enable dKV-Cache for faster inference
            cache_reloading_step: Reload cache every N steps (dKV-Cache-Decode)

        Returns:
            Generated token ids (batch_size, seq_len)
        """
        if use_dkv_cache:
            return self._sample_with_dkv_cache(
                model, batch_size, seq_len, steps, temperature,
                device, prompt, strategy, cache_reloading_step
            )

        # Original sampling without cache
        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        prompt_len = 0
        if prompt is not None:
            prompt_len = prompt.shape[1]
            x[:, :prompt_len] = prompt

        timesteps = torch.linspace(1.0, self.eps, steps + 1, device=device)

        # Compute transfer schedule for confidence strategy (same as dKV-cache)
        if strategy == "confidence":
            init_mask_index = (x == self.mask_token_id)
            num_transfer_tokens = self._get_num_transfer_tokens(init_mask_index, steps)

        for i in range(steps):
            t, s = timesteps[i].item(), timesteps[i + 1].item()
            mask_indices = (x == self.mask_token_id)
            if prompt_len > 0:
                mask_indices[:, :prompt_len] = False
            if not mask_indices.any():
                break

            logits, _ = model(x)

            if strategy == "confidence":
                # Use same Gumbel-max sampling as dKV-cache for fair comparison
                logits_with_noise = self._add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Compute confidence
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

                # Only consider masked positions
                x0 = torch.where(mask_indices, x0, x)
                confidence = torch.where(mask_indices, x0_p, torch.tensor(-np.inf, device=device))

                # Protect prompt positions
                if prompt_len > 0:
                    confidence[:, :prompt_len] = -np.inf

                # Select top-k confident positions to unmask
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                for j in range(batch_size):
                    k = num_transfer_tokens[j, i].item()
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Update x with newly decoded tokens
                x[transfer_index] = x0[transfer_index]
            else:
                # Stochastic unmasking (original behavior)
                masked_logits = logits[mask_indices] / temperature
                probs = F.softmax(masked_logits, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)

                p_transfer = 1 - s / t if t > 0 else 1.0
                unmask = torch.rand(sampled.shape, device=device) < p_transfer
                mask_pos = mask_indices.nonzero(as_tuple=False)
                for j, (b, pos) in enumerate(mask_pos):
                    if unmask[j]:
                        x[b, pos] = sampled[j]
        return x

    @torch.no_grad()
    def _sample_with_dkv_cache(
        self,
        model,
        batch_size: int,
        seq_len: int,
        steps: int,
        temperature: float,
        device: str,
        prompt: Optional[torch.Tensor],
        strategy: str,
        cache_reloading_step: int,
    ) -> torch.Tensor:
        """
        dKV-Cache enabled sampling (dKV-Cache-Decode variant).

        This implements the near-lossless dKV-Cache-Decode strategy from the paper:
        - Cache K,V for decoded (unmasked) tokens
        - Reload cache periodically (cache_reloading_step) to maintain quality
        - Use confidence-based unmasking (low_confidence remasking)

        Reference: https://github.com/horseee/dKV-Cache
        """
        B = batch_size
        x = torch.full((B, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        prompt_len = 0
        if prompt is not None:
            prompt_len = prompt.shape[1]
            x[:, :prompt_len] = prompt

        # Initialize cache
        model.init_dkv_cache(torch.device(device))

        # Track decoded positions
        prv_decoded_mask: Optional[torch.Tensor] = None
        cur_decoded_mask: Optional[torch.Tensor] = None

        # Compute transfer schedule
        init_mask_index = (x == self.mask_token_id)
        num_transfer_tokens = self._get_num_transfer_tokens(init_mask_index, steps)

        for i in range(steps):
            mask_index = (x == self.mask_token_id)

            # Determine cache mode for this step
            use_cache_this_step = (i > 0) and (i % cache_reloading_step != 0)
            reload_cache = (i > 0) and (i % cache_reloading_step == 0)

            if use_cache_this_step and prv_decoded_mask is not None:
                # Forward with cache - reuse cached K,V for decoded tokens
                logits, _ = model(
                    x,
                    cache_position=cur_decoded_mask,
                    prv_cache_position=prv_decoded_mask,
                    use_cache=True,
                )
            elif reload_cache:
                # Reload cache - full forward pass, update cache
                model.reset_dkv_cache()
                logits, _ = model(
                    x,
                    cache_position=cur_decoded_mask,
                    prv_cache_position=None,
                    use_cache=False,
                )
            else:
                # First step - no cache
                logits, _ = model(x, use_cache=False)

            # Apply Gumbel noise and get predictions
            logits_with_noise = self._add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute confidence for remasking (dKV-Cache always uses confidence-based)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # Only consider masked positions
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device))

            # Protect prompt positions
            if prompt_len > 0:
                confidence[:, :prompt_len] = -np.inf

            # Select top-k confident positions to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(B):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_idx = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_idx] = True

            # Update x with newly decoded tokens
            x[transfer_index] = x0[transfer_index]

            # Update decoded masks for next iteration
            prv_decoded_mask = cur_decoded_mask
            cur_decoded_mask = (x != self.mask_token_id)

        # Cleanup
        model.disable_dkv_cache()

        return x
