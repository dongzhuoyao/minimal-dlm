"""
Masked Diffusion Process - Forward masking and reverse unmasking for discrete tokens.
"""
import torch
import torch.nn.functional as F
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

    @torch.no_grad()
    def sample(self, model, batch_size: int, seq_len: int, steps: int = 64,
               temperature: float = 1.0, device: str = "cuda",
               prompt: Optional[torch.Tensor] = None, strategy: str = "stochastic"
               ) -> torch.Tensor:
        """Generate by iterative unmasking."""
        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        prompt_len = 0
        if prompt is not None:
            prompt_len = prompt.shape[1]
            x[:, :prompt_len] = prompt

        timesteps = torch.linspace(1.0, self.eps, steps + 1, device=device)

        for i in range(steps):
            t, s = timesteps[i].item(), timesteps[i + 1].item()
            mask_indices = (x == self.mask_token_id)
            if prompt_len > 0:
                mask_indices[:, :prompt_len] = False
            if not mask_indices.any():
                break

            logits, _ = model(x)
            masked_logits = logits[mask_indices] / temperature
            probs = F.softmax(masked_logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            if strategy == "confidence":
                # Unmask highest confidence predictions
                p_transfer = 1 - s / t if t > 0 else 1.0
                num_to_unmask = max(1, int(mask_indices.sum().item() * p_transfer))
                confidence = torch.gather(probs, 1, sampled.unsqueeze(-1)).squeeze(-1)
                _, top_idx = torch.topk(confidence, min(num_to_unmask, len(confidence)))
                mask_pos = mask_indices.nonzero(as_tuple=False)
                for idx in top_idx:
                    b, pos = mask_pos[idx]
                    x[b, pos] = sampled[idx]
            else:
                # Stochastic unmasking
                p_transfer = 1 - s / t if t > 0 else 1.0
                unmask = torch.rand(sampled.shape, device=device) < p_transfer
                mask_pos = mask_indices.nonzero(as_tuple=False)
                for j, (b, pos) in enumerate(mask_pos):
                    if unmask[j]:
                        x[b, pos] = sampled[j]
        return x
