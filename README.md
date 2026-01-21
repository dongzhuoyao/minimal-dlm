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

## Dataset

### Shakespeare Character-Level

The default dataset is **Tiny Shakespeare** (~1MB, 1.1M characters), automatically downloaded on first run.

| Split | Characters | Tokens |
|-------|-----------|--------|
| Train | ~1M | ~1M |
| Val | ~100K | ~100K |
| Vocab | 65 unique characters | - |

**Data format** (nanoGPT-style):
- `train.bin`, `val.bin`: Binary files of `uint16` token IDs
- `meta.pkl`: Vocabulary metadata (`stoi`, `itos`, `vocab_size`)

**Preparation** (automatic or manual):
```bash
# Auto-downloads and prepares data on first train.py run
# Or manually:
python -c "from data import prepare_shakespeare_char; prepare_shakespeare_char('data/shakespeare_char')"
```

### Adding Custom Datasets

To use your own data, create `train.bin` and `val.bin` with `uint16` token IDs:

```python
import numpy as np
import pickle

# Your tokenized data
train_ids = [...]  # list of token IDs
val_ids = [...]

# Save binary files
np.array(train_ids, dtype=np.uint16).tofile('data/mydata/train.bin')
np.array(val_ids, dtype=np.uint16).tofile('data/mydata/val.bin')

# Save metadata
meta = {'vocab_size': YOUR_VOCAB_SIZE, 'stoi': {...}, 'itos': {...}}
with open('data/mydata/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
```

Then train with: `python train.py data.dir=data/mydata`

## Evaluation

### Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Loss** | Weighted cross-entropy on masked tokens | `CE / p_mask` |
| **Perplexity (PPL)** | Exponentiated loss (branching factor) | `exp(loss)` |
| **Bits per Character (BPC)** | Information-theoretic measure | `loss / ln(2)` |

### Monte Carlo Estimation

Since MDMs use random masking, evaluation uses **Monte Carlo sampling**:
- Average loss over `mc_samples` different random maskings per batch
- Default: 16 MC samples × 20 eval batches = 320 forward passes

```bash
# Standard evaluation
python eval.py checkpoint=out/ckpt.pt

# More accurate (more MC samples)
python eval.py checkpoint=out/ckpt.pt mc_samples=64 eval_iters=50

# Evaluate on train split
python eval.py checkpoint=out/ckpt.pt split=train
```

### Expected Results (Shakespeare char-level)

| Model | Params | Val Loss | Val PPL | Val BPC |
|-------|--------|----------|---------|---------|
| 4L-4H-256D | ~2.5M | ~1.5 | ~4.5 | ~2.2 |
| 6L-6H-384D | ~10M | ~1.3 | ~3.7 | ~1.9 |

*Results vary with training iterations and hyperparameters.*

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
