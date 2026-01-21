"""
Training script for Minimal Masked Diffusion Language Model.

Usage:
    python train.py                                    # Default config
    python train.py training.max_iters=10000           # Override iterations
    python train.py +experiment=medium                 # Use medium model config
    python train.py wandb.enabled=true                 # Enable wandb
"""
import os
import math
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from model import MaskedDiffusionTransformer, ModelConfig
from diffusion import MaskedDiffusion
from data import get_batch, load_meta, prepare_shakespeare_char


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, diffusion, data_dir, block_size, batch_size, eval_iters, device, ctx):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        total = 0.0
        for _ in range(eval_iters):
            x, _ = get_batch(split, data_dir, batch_size, block_size, device)
            x_t, mask_indices, p_mask = diffusion.forward_process(x)
            with ctx:
                _, loss = model(x_t, x, mask_indices, p_mask)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses


@torch.no_grad()
def generate_samples(model, diffusion, dataset, device, num_samples=3, seq_len=100, steps=32):
    """Generate sample text for wandb logging."""
    model.eval()
    samples = []
    for _ in range(num_samples):
        output = diffusion.sample(model, 1, seq_len, steps=steps, device=device)
        ids = [t for t in output[0].tolist() if t < diffusion.vocab_size]
        samples.append(dataset.decode(ids))
    model.train()
    return samples


def init_wandb(cfg: DictConfig, model, output_dir: Path):
    """Initialize wandb with organized config."""
    import wandb

    # Generate run name if not provided
    run_name = cfg.wandb.name
    if run_name is None:
        run_name = output_dir.name

    # Flatten config for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=config_dict,
        dir=str(output_dir),
        reinit=True,
    )

    # Log model architecture
    wandb.config.update({
        "model/num_params": sum(p.numel() for p in model.parameters()),
        "model/num_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    })

    # Define custom metrics with step as x-axis
    wandb.define_metric("step")
    wandb.define_metric("loss/*", step_metric="step")
    wandb.define_metric("perf/*", step_metric="step")
    wandb.define_metric("optim/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")

    return wandb


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Setup
    torch.manual_seed(cfg.system.seed)
    device = cfg.system.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.system.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else nullcontext()

    # Get directories
    orig_cwd = hydra.utils.get_original_cwd()
    data_dir = os.path.join(orig_cwd, cfg.data.dir)

    # Use Hydra output directory for checkpoints
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        prepare_shakespeare_char(data_dir)
    meta = load_meta(data_dir)
    vocab_size = meta["vocab_size"]

    # Create model
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.model.block_size,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    )
    model = MaskedDiffusionTransformer(model_cfg).to(device)
    diffusion = MaskedDiffusion(vocab_size, eps=cfg.diffusion.eps)

    # Compile model if requested
    if cfg.system.compile and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    # Initialize wandb
    wandb = None
    if cfg.wandb.enabled:
        wandb = init_wandb(cfg, model, output_dir)

        # Watch model gradients
        wandb.watch(model, log="gradients", log_freq=100)

    # Optimizer
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay,
        cfg.training.learning_rate,
        (0.9, 0.95),
        device_type,
    )
    scaler = torch.amp.GradScaler(enabled=(cfg.system.dtype == "float16"))

    # Training state
    iter_num, best_val_loss = 0, float("inf")

    # Load dataset for sample generation
    from data import TextDataset
    dataset = TextDataset(data_dir)

    print(f"Training for {cfg.training.max_iters} iterations")
    print(f"Output: {output_dir}")

    t0 = time.time()
    running_loss = 0.0

    while iter_num < cfg.training.max_iters:
        lr = get_lr(iter_num, cfg.training.warmup_iters, cfg.training.max_iters,
                    cfg.training.learning_rate, cfg.training.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Evaluate
        if iter_num % cfg.training.eval_interval == 0:
            losses = estimate_loss(model, diffusion, data_dir, cfg.model.block_size,
                                   cfg.training.batch_size, cfg.training.eval_iters, device, ctx)
            print(f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            # Log to wandb
            if wandb:
                log_dict = {
                    "step": iter_num,
                    "eval/train_loss": losses["train"],
                    "eval/val_loss": losses["val"],
                    "eval/train_ppl": math.exp(losses["train"]),
                    "eval/val_ppl": math.exp(losses["val"]),
                    "eval/train_bpc": losses["train"] / math.log(2),
                    "eval/val_bpc": losses["val"] / math.log(2),
                }

                # Generate and log samples
                if iter_num % (cfg.training.eval_interval * 4) == 0:
                    samples = generate_samples(model, diffusion, dataset, device)
                    log_dict["samples"] = wandb.Table(
                        columns=["step", "sample"],
                        data=[[iter_num, s] for s in samples]
                    )

                wandb.log(log_dict)

            # Save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": model_cfg.__dict__,
                    "hydra_cfg": OmegaConf.to_container(cfg),
                }
                ckpt_path = checkpoint_dir / "ckpt.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Train step
        x, _ = get_batch("train", data_dir, cfg.training.batch_size, cfg.model.block_size, device)
        x_t, mask_indices, p_mask = diffusion.forward_process(x)

        with ctx:
            _, loss = model(x_t, x, mask_indices, p_mask)

        scaler.scale(loss).backward()

        if cfg.training.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        else:
            grad_norm = 0.0

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()

        # Logging
        if iter_num % cfg.training.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_loss = running_loss / cfg.training.log_interval if iter_num > 0 else loss.item()
            running_loss = 0.0

            # Calculate throughput
            iter_per_sec = cfg.training.log_interval / dt if dt > 0 else 0
            tokens_per_sec = iter_per_sec * cfg.training.batch_size * cfg.model.block_size

            print(f"iter {iter_num}: loss {loss.item():.4f}, {iter_per_sec:.1f} it/s, lr {lr:.2e}")

            if wandb:
                wandb.log({
                    "step": iter_num,
                    # Loss metrics
                    "loss/train": loss.item(),
                    "loss/train_avg": avg_loss,
                    # Performance metrics
                    "perf/iter_per_sec": iter_per_sec,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/time_ms": dt * 1000,
                    # Optimizer metrics
                    "optim/lr": lr,
                    "optim/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                })

        iter_num += 1

    print("Training complete!")

    if wandb:
        # Log final samples
        samples = generate_samples(model, diffusion, dataset, device, num_samples=5, seq_len=200)
        wandb.log({
            "final_samples": wandb.Table(
                columns=["sample"],
                data=[[s] for s in samples]
            )
        })
        wandb.finish()


if __name__ == "__main__":
    main()
