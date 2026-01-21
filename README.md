# Minimal Masked Diffusion Language Model

A clean, minimal implementation of **Masked Diffusion Models** for language modeling.

## Files

```
model.py              # Bidirectional transformer (RoPE, RMSNorm, SwiGLU)
diffusion.py          # Forward masking + reverse sampling
data.py               # Data loading (nanoGPT-style binary format)
train.py              # Training script (Hydra + wandb)
sample.py             # Text generation
eval.py               # Evaluation (loss, perplexity)
configs/
  config.yaml         # Default configuration
  experiment/         # Experiment configs (small, medium)
```

## Quick Start

```bash
# Install dependencies
pip install torch numpy hydra-core omegaconf wandb

# Train (auto-downloads Shakespeare)
python train.py

# Generate text
python sample.py checkpoint=out/ckpt.pt prompt="ROMEO:"

# Evaluate
python eval.py checkpoint=out/ckpt.pt
```

## Configuration (Hydra)

All scripts use [Hydra](https://hydra.cc/) for configuration management.

```bash
# Override any config parameter
python train.py training.max_iters=10000 model.n_layer=6

# Use experiment preset
python train.py +experiment=medium

# Enable wandb logging
python train.py logging.wandb=true logging.project=my-project

# Multiple overrides
python train.py model.n_layer=6 model.n_head=6 model.n_embd=384 \
                training.learning_rate=6e-4 logging.wandb=true
```

## Weights & Biases Integration

Organized wandb logging with:
- **train/**: loss, loss_avg, lr, grad_norm, time_ms
- **eval/**: train_loss, val_loss, train_ppl, val_ppl
- **samples**: Generated text samples (logged periodically)
- **Config**: Full Hydra config logged automatically
- **Model**: Parameter count, architecture details

```bash
# Enable wandb
python train.py logging.wandb=true

# Custom project/run name
python train.py logging.wandb=true logging.project=mdm-experiments \
                logging.name=my-run logging.tags=[small,debug]
```

## Core Concepts

### Forward Process (Masking)
```
p_mask(t) = (1 - ε) * t + ε    # ε=0.001
```
- `t=0`: Almost no masking
- `t=1`: Complete masking

### Training Loss
```
L = E[CE(model(x_masked), x_original) / p_mask]
```
Importance weighting by `1/p_mask` ensures proper likelihood estimation.

### Generation (Reverse Process)
1. Start fully masked
2. Model predicts all positions
3. Unmask fraction `1 - s/t` each step
4. Repeat until clean

**Strategies**: `stochastic` (random) or `confidence` (highest confidence first)

## Architecture

- **Bidirectional attention** (no causal mask)
- **RMSNorm** + **RoPE** + **SwiGLU**
- Mask token = `vocab_size` (appended to vocabulary)

## Training Examples

```bash
# Default small model
python train.py

# Medium model with wandb
python train.py +experiment=medium logging.wandb=true

# Custom model
python train.py model.n_layer=8 model.n_head=8 model.n_embd=512 \
                training.max_iters=20000

# CPU training
python train.py system.device=cpu system.dtype=float32
```

## Sampling Examples

```bash
# Basic sampling
python sample.py checkpoint=out/ckpt.pt

# With prompt
python sample.py checkpoint=out/ckpt.pt prompt="To be or not"

# Adjust quality/diversity
python sample.py checkpoint=out/ckpt.pt \
                sampling.steps=128 sampling.temperature=0.8

# Confidence-based sampling
python sample.py checkpoint=out/ckpt.pt sampling.strategy=confidence

# Multiple samples
python sample.py checkpoint=out/ckpt.pt num_samples=5
```

## References

- [SMDM](https://arxiv.org/abs/2410.18514): Scaling up Masked Diffusion Models
- [D3PM](https://arxiv.org/abs/2107.03006): Discrete Denoising Diffusion
- [nanoGPT](https://github.com/karpathy/nanoGPT): Minimal GPT
