"""
Sampling script for Minimal Masked Diffusion Language Model.

Usage:
    python sample.py checkpoint=out/ckpt.pt
    python sample.py checkpoint=out/ckpt.pt prompt="ROMEO:"
    python sample.py checkpoint=out/ckpt.pt sampling.steps=128 sampling.temperature=0.8
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from model import MaskedDiffusionTransformer, ModelConfig
from diffusion import MaskedDiffusion
from data import TextDataset


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Get checkpoint path
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        print("Usage: python sample.py checkpoint=out/ckpt.pt")
        return

    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint = os.path.join(orig_cwd, checkpoint)

    # Setup device
    device = cfg.system.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if cfg.system.seed:
        torch.manual_seed(cfg.system.seed)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"])
    model = MaskedDiffusionTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Get diffusion eps from checkpoint if available
    eps = 1e-3
    if "hydra_cfg" in ckpt:
        eps = ckpt["hydra_cfg"].get("diffusion", {}).get("eps", 1e-3)
    diffusion = MaskedDiffusion(model_cfg.vocab_size, eps=eps)

    # Load dataset
    data_dir = os.path.join(orig_cwd, cfg.data.dir)
    try:
        dataset = TextDataset(data_dir)
    except:
        dataset = None
        print("Warning: Could not load dataset for decoding")

    # Get sampling params (allow override from command line)
    prompt = cfg.get("prompt", None)
    num_samples = cfg.get("num_samples", 1)
    max_tokens = cfg.get("max_tokens", 200)
    steps = cfg.sampling.steps
    temperature = cfg.sampling.temperature
    strategy = cfg.sampling.strategy

    # dKV-Cache settings
    use_dkv = cfg.diffusion.get("use_dkv", False)
    cache_reloading_step = cfg.diffusion.get("cache_reloading_step", 1)

    # Prepare prompt
    prompt_tensor, prompt_len = None, 0
    if prompt and dataset:
        prompt_ids = dataset.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)
        print(f"Prompt: '{prompt}' ({prompt_len} tokens)")

    seq_len = min(prompt_len + max_tokens, model_cfg.block_size)
    print(f"Generating {num_samples} sample(s), {seq_len} tokens, {steps} steps")
    print(f"Temperature: {temperature}, Strategy: {strategy}")
    if use_dkv:
        print(f"dKV-Cache: enabled (reload every {cache_reloading_step} steps)")
    print()

    for i in range(num_samples):
        if num_samples > 1:
            print(f"--- Sample {i+1} ---")

        output = diffusion.sample(
            model, batch_size=1, seq_len=seq_len, steps=steps,
            temperature=temperature, device=device,
            prompt=prompt_tensor, strategy=strategy,
            use_dkv_cache=use_dkv, cache_reloading_step=cache_reloading_step
        )

        if dataset:
            ids = [t for t in output[0].tolist() if t != model_cfg.mask_token_id and t < model_cfg.vocab_size]
            print(dataset.decode(ids))
        else:
            print(output[0].tolist())
        print()


if __name__ == "__main__":
    main()
