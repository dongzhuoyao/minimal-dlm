"""
DUO (Discrete Uniform diffusion with One-hot encoding) - Extracted from original implementation.

Key differences from MaskedDiffusion (MDM):
- Forward: Gaussian noise on one-hot encodings -> discretize via argmax
- Loss: Complex posterior-based likelihood with dalpha_t weighting
- Sampling: Ancestral sampling from exact posterior q(x_s|x_t, x_0)
- Training: Curriculum learning with Gumbel-softmax temperature annealing

Reference: original_impl/duo-main/algo.py
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class LogLinearSchedule:
    """Log-linear noise schedule: alpha_t = 1 - t, consistent with SEDD."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def __call__(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (dalpha_t, alpha_t) for given t."""
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = -(1 - self.eps)
        return dalpha_t, alpha_t


class DUO:
    """
    DUO: Discrete Uniform diffusion with One-hot encoding.

    Forward: q(x_t | x_0) uses Gaussian noise on one-hot vectors, discretized via argmax.
    Loss: Derived from continuous diffusion theory with posterior-based likelihood.

    Key parameters:
    - vocab_size: Size of vocabulary (without mask token for uniform state)
    - eps: Minimum noise level (sampling_eps)
    - curriculum_start/end: Global steps for curriculum learning phase
    - gumbel_tau_log10_start/end: Temperature annealing range (log10 scale)
    - gamma_min/max: Gamma range for Gaussian forward process
    """

    def __init__(
        self,
        vocab_size: int,
        eps: float = 1e-3,
        gamma_min: float = -10.0,
        gamma_max: float = 10.0,
        curriculum_start: int = 0,
        curriculum_end: int = 5000,
        gumbel_tau_log10_start: float = 0.0,
        gumbel_tau_log10_end: float = -10.0,
    ):
        self.vocab_size = vocab_size
        self.eps = eps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end
        self.gumbel_tau_log10_start = gumbel_tau_log10_start
        self.gumbel_tau_log10_end = gumbel_tau_log10_end

        # Log-linear schedule for alpha_t
        self.noise = LogLinearSchedule(eps)

    def _compute_gumbel_tau_inverse(self, global_step: int) -> float:
        """Compute inverse Gumbel temperature for curriculum learning."""
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start

        if global_step < self.curriculum_start:
            tau = start
        elif global_step < self.curriculum_end:
            frac = (global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = -10  # Very low temperature (hard discretization)

        return 10 ** (-tau)

    def _sigma_from_alphat(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Convert alpha_t to sigma (log-space)."""
        return -torch.log(alpha_t.clamp(min=1e-10))

    def _q_xt_gaussian(
        self,
        x0_onehot: torch.Tensor,
        gamma_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute noisy sample in continuous space using Gaussian noise.

        Args:
            x0_onehot: One-hot encoded tokens (B, T, V)
            gamma_t: Gamma values per batch (B,)

        Returns:
            Noisy continuous representation (B, T, V)
        """
        assert gamma_t.ndim == 1
        assert x0_onehot.ndim == 3

        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        epsilon = torch.randn_like(x0_onehot)
        return alpha_t * x0_onehot + sigma_t * epsilon

    def _gamma_to_alphat_simple(self, gamma_t: torch.Tensor) -> torch.Tensor:
        """
        Simple approximation: convert gamma to alpha_t.

        In the original DUO, this uses a precomputed integral cache.
        Here we use a simplified version based on the log-linear schedule.
        """
        # Normalize gamma to [0, 1] range
        t = (gamma_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t = t.clamp(0, 1)
        # Apply log-linear schedule
        _, alpha_t = self.noise(t)
        return alpha_t

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion using Gaussian noise on one-hot encodings.

        Args:
            x_0: Original token indices (B, T)
            t: Optional timesteps (B,), sampled uniformly if None
            global_step: Current training step for curriculum learning

        Returns:
            x_t: Noised discrete tokens (B, T)
            alpha_t: Signal level (B, 1)
            dalpha_t: Time derivative of alpha (B, 1) or scalar
            is_curriculum: Whether in curriculum phase
        """
        B, T = x_0.shape
        device = x_0.device

        if t is None:
            t = torch.rand(B, device=device)

        # Curriculum phase check
        is_curriculum = global_step < self.curriculum_end

        # Compute gamma and alpha
        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        alpha_t = self._gamma_to_alphat_simple(gamma_t)

        # Compute dalpha_t (time derivative approximation)
        delta = 1e-3
        gamma_t_plus = gamma_t + delta * (self.gamma_max - self.gamma_min)
        alpha_t_plus = self._gamma_to_alphat_simple(gamma_t_plus)
        dalpha_t = (alpha_t_plus - alpha_t) / delta

        alpha_t = alpha_t.unsqueeze(-1)  # (B, 1)
        dalpha_t = dalpha_t.unsqueeze(-1)  # (B, 1)

        if is_curriculum:
            # Curriculum phase: use Gaussian noise + Gumbel temperature
            x0_onehot = F.one_hot(x_0, self.vocab_size).float()
            xt_continuous = self._q_xt_gaussian(x0_onehot, gamma_t)

            # Apply Gumbel temperature
            tau_inv = self._compute_gumbel_tau_inverse(global_step)
            xt_continuous = xt_continuous * tau_inv

            # Discretize via argmax
            x_t = xt_continuous.argmax(dim=-1)
        else:
            # Post-curriculum: use discrete uniform noise
            move_indices = torch.rand(B, T, device=device) < (1 - alpha_t)
            uniform_tokens = torch.randint(0, self.vocab_size, (B, T), device=device)
            x_t = torch.where(move_indices, uniform_tokens, x_0)

        return x_t, alpha_t, dalpha_t, is_curriculum

    def compute_loss(
        self,
        logits: torch.Tensor,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        alpha_t: torch.Tensor,
        dalpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DUO loss (ELBO-derived).

        This is the core nll_per_token from original DUO implementation.

        Args:
            logits: Model output logits (B, T, V)
            x_t: Noised tokens (B, T)
            x_0: Original tokens (B, T)
            alpha_t: Signal level (B, 1)
            dalpha_t: Time derivative (B, 1)

        Returns:
            Loss tensor (scalar)
        """
        B, T, V = logits.shape

        # Normalize to log probabilities
        log_x_theta = F.log_softmax(logits, dim=-1)
        x_reconst = log_x_theta.exp()

        # Compute x_bar_theta = V * alpha_t * x_theta + (1 - alpha_t)
        x_bar_theta = (self.vocab_size * alpha_t.unsqueeze(-1) * x_reconst
                       + 1 - alpha_t.unsqueeze(-1))

        # Coefficient for the loss
        coeff = dalpha_t / (self.vocab_size * alpha_t + 1e-10)

        # Masks for x_0 == x_t and x_0 != x_t
        x_eq_xt = (x_0 == x_t).float()
        x_neq_xt = 1 - x_eq_xt

        # x_bar at x_t position
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt

        # Gather x_bar_theta at x_t and x_0 positions
        xbar_theta_xt = torch.gather(x_bar_theta, -1, x_t.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x0 = torch.gather(x_bar_theta, -1, x_0.unsqueeze(-1)).squeeze(-1)

        # Term 1: V * (1/xbar_xt - 1/xbar_theta_xt)
        term1 = self.vocab_size * (1 / (xbar_xt + 1e-10) - 1 / (xbar_theta_xt + 1e-10))

        # Term 2 computation (complex, from original DUO)
        const = (1 - alpha_t) / (self.vocab_size * alpha_t + 1 - alpha_t + 1e-10)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = (
            (self.vocab_size - 1) * const * x_eq_xt
            - (1 / (const + 1e-10)) * x_neq_xt
        ) * (const + 1e-10).log()

        term2_theta = -term2_coefs * (
            x_bar_theta.log().sum(-1)
            - self.vocab_size * (xbar_theta_xt + 1e-10).log()
        )
        term2_theta = term2_theta - (
            self.vocab_size * alpha_t / (1 - alpha_t + 1e-10)
            * ((xbar_theta_x0 + 1e-10).log() - (xbar_theta_xt + 1e-10).log())
            * x_neq_xt
        )
        term2 = term2_theta + term2_offset

        # Diffusion loss
        diffusion_loss = coeff * (term1 - term2)

        # Average over batch and sequence
        loss = diffusion_loss.mean()

        return loss

    def _compute_posterior(
        self,
        x_theta: torch.Tensor,
        x_t: torch.Tensor,
        alpha_s: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute posterior q(x_s | x_t, x_0) for sampling.

        Args:
            x_theta: Predicted x_0 distribution (B, T, V)
            x_t: Current noised tokens (B, T)
            alpha_s: Signal level at s (B, 1) or (B, T, 1)
            alpha_t: Signal level at t (B, 1) or (B, T, 1)

        Returns:
            Posterior probabilities (B, T, V)
        """
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)

        alpha_ts = alpha_t / (alpha_s + 1e-10)
        d_alpha = alpha_s - alpha_t

        xt_onehot = F.one_hot(x_t, self.vocab_size).float()

        # Posterior computation from original DUO
        numerator = (
            alpha_t * self.vocab_size * x_theta * xt_onehot
            + (alpha_ts - alpha_t) * xt_onehot
            + d_alpha * x_theta
            + (1 - alpha_ts) * (1 - alpha_s) / self.vocab_size
        )

        x_theta_at_xt = torch.gather(x_theta, -1, x_t.unsqueeze(-1))
        denominator = alpha_t * self.vocab_size * x_theta_at_xt + (1 - alpha_t)

        posterior = numerator / (denominator + 1e-10)

        return posterior

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
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using ancestral sampling from posterior.

        Args:
            model: The transformer model
            batch_size: Number of sequences to generate
            seq_len: Length of each sequence
            steps: Number of diffusion steps
            temperature: Sampling temperature
            device: Device to use
            prompt: Optional prompt tokens
            top_p: Nucleus sampling probability (1.0 = no filtering)

        Returns:
            Generated token indices (B, T)
        """
        # Initialize with uniform random tokens
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)

        prompt_len = 0
        if prompt is not None:
            prompt_len = prompt.shape[1]
            x[:, :prompt_len] = prompt

        timesteps = torch.linspace(1.0, self.eps, steps + 1, device=device)

        for i in range(steps):
            t = timesteps[i]
            s = timesteps[i + 1]

            # Compute alpha values
            _, alpha_t = self.noise(t)
            _, alpha_s = self.noise(s)

            alpha_t = alpha_t.unsqueeze(0).expand(batch_size, 1)
            alpha_s = alpha_s.unsqueeze(0).expand(batch_size, 1)

            # Get model predictions
            sigma_t = self._sigma_from_alphat(alpha_t)
            logits, _ = model(x)
            logits = logits / temperature
            x_theta = F.softmax(logits, dim=-1)

            # Compute posterior
            q_xs = self._compute_posterior(x_theta, x, alpha_s, alpha_t)

            # Optional nucleus sampling
            if top_p < 1.0:
                q_xs = self._nucleus_filter(q_xs, top_p)

            # Sample from posterior using Gumbel-max trick
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(q_xs) + 1e-10) + 1e-10)
            x_new = (q_xs.log() + gumbel_noise).argmax(dim=-1)

            # Preserve prompt
            if prompt_len > 0:
                x_new[:, :prompt_len] = prompt

            x = x_new

        # Final denoising step
        logits, _ = model(x)
        x = logits.argmax(dim=-1)

        if prompt_len > 0:
            x[:, :prompt_len] = prompt

        return x

    def _nucleus_filter(
        self,
        probs: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to probabilities."""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs = probs.masked_fill(indices_to_remove, 0.0)

        # Renormalize
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        return probs
