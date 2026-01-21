"""
Evaluation script for Minimal Masked Diffusion Language Model.

Usage:
    python eval.py checkpoint=out/ckpt.pt
    python eval.py checkpoint=out/ckpt.pt eval_iters=100 mc_samples=32
    python eval.py checkpoint=out/ckpt.pt wandb.enabled=true
"""
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

from model import MaskedDiffusionTransformer, ModelConfig
from diffusion import MaskedDiffusion
from data import get_batch


@torch.no_grad()
def evaluate(model, diffusion, data_dir, block_size, batch_size, eval_iters, mc_samples, device):
    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, _ = get_batch(split, data_dir, batch_size, block_size, device)
            batch_losses = []
            for _ in range(mc_samples):
                x_t, mask_indices, p_mask = diffusion.forward_process(x)
                _, loss = model(x_t, x, mask_indices, p_mask)
                batch_losses.append(loss.item())
            losses.append(np.mean(batch_losses))

        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        results[f"{split}_loss"] = avg_loss
        results[f"{split}_loss_std"] = std_loss
        results[f"{split}_perplexity"] = np.exp(avg_loss)
        results[f"{split}_bpc"] = avg_loss / np.log(2)

    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Get checkpoint path
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        print("Usage: python eval.py checkpoint=out/ckpt.pt")
        return

    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint = os.path.join(orig_cwd, checkpoint)
    data_dir = os.path.join(orig_cwd, cfg.data.dir)

    # Setup device
    device = cfg.system.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Get eval params (allow override)
    eval_iters = cfg.get("eval_iters", cfg.training.eval_iters)
    mc_samples = cfg.get("mc_samples", 16)
    batch_size = cfg.get("batch_size", cfg.training.batch_size)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"])
    model = MaskedDiffusionTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    # Get diffusion eps from checkpoint if available
    eps = 1e-3
    if "hydra_cfg" in ckpt:
        eps = ckpt["hydra_cfg"].get("diffusion", {}).get("eps", 1e-3)
    diffusion = MaskedDiffusion(model_cfg.vocab_size, eps=eps)

    print(f"Model: {model_cfg.n_layer}L-{model_cfg.n_head}H-{model_cfg.n_embd}D")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Checkpoint iter: {ckpt.get('iter_num', '?')}")
    print(f"Best val loss: {ckpt.get('best_val_loss', '?')}")
    print(f"\nEvaluating with {eval_iters} iters, {mc_samples} MC samples...")

    results = evaluate(model, diffusion, data_dir, model_cfg.block_size,
                       batch_size, eval_iters, mc_samples, device)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in results.items():
        print(f"{k:20s}: {v:.4f}")
    print("=" * 50)

    # Log to wandb if enabled
    if cfg.wandb.enabled:
        import wandb

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        run_name = f"eval_{output_dir.name}"

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(output_dir),
            job_type="eval",
        )

        # Log model info
        wandb.config.update({
            "model/architecture": f"{model_cfg.n_layer}L-{model_cfg.n_head}H-{model_cfg.n_embd}D",
            "model/num_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "eval/checkpoint": checkpoint,
            "eval/checkpoint_iter": ckpt.get("iter_num", -1),
            "eval/eval_iters": eval_iters,
            "eval/mc_samples": mc_samples,
        })

        # Log results
        wandb.log({
            "eval/train_loss": results["train_loss"],
            "eval/val_loss": results["val_loss"],
            "eval/train_loss_std": results["train_loss_std"],
            "eval/val_loss_std": results["val_loss_std"],
            "eval/train_ppl": results["train_perplexity"],
            "eval/val_ppl": results["val_perplexity"],
            "eval/train_bpc": results["train_bpc"],
            "eval/val_bpc": results["val_bpc"],
        })

        # Log summary table
        wandb.log({
            "results": wandb.Table(
                columns=["metric", "value"],
                data=[[k, v] for k, v in results.items()]
            )
        })

        wandb.finish()
        print(f"\nResults logged to wandb project: {cfg.wandb.project}")


if __name__ == "__main__":
    main()
