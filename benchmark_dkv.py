"""
Benchmark dKV-Cache performance for Masked Diffusion Models.

Compares inference speed and generation quality with/without dKV-Cache.

Usage:
    python benchmark_dkv.py checkpoint=outputs/checkpoints/ckpt.pt
    python benchmark_dkv.py checkpoint=out/ckpt.pt num_runs=10 seq_len=128
    python benchmark_dkv.py checkpoint=out/ckpt.pt cache_reloading_steps=[1,2,4,8]
    python benchmark_dkv.py checkpoint=out/ckpt.pt wandb.enabled=true
"""
import os
import time
from contextlib import contextmanager
from typing import List, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from model import MaskedDiffusionTransformer, ModelConfig
from diffusion import MaskedDiffusion
from data import TextDataset


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def measure_sampling_time(
    model: MaskedDiffusionTransformer,
    diffusion: MaskedDiffusion,
    batch_size: int,
    seq_len: int,
    steps: int,
    device: str,
    use_dkv: bool,
    cache_reloading_step: int = 1,
    num_runs: int = 5,
    warmup_runs: int = 2,
    strategy: str = "confidence",
) -> Dict[str, float]:
    """
    Measure sampling time with multiple runs.

    Returns dict with mean, std, min, max times in seconds.
    """
    times = []

    # Warmup runs (not counted)
    for _ in range(warmup_runs):
        _ = diffusion.sample(
            model, batch_size, seq_len, steps=steps, device=device,
            strategy=strategy,
            use_dkv_cache=use_dkv, cache_reloading_step=cache_reloading_step
        )
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = diffusion.sample(
            model, batch_size, seq_len, steps=steps, device=device,
            strategy=strategy,
            use_dkv_cache=use_dkv, cache_reloading_step=cache_reloading_step
        )

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "times": times,
    }


def compute_output_similarity(
    outputs_baseline: List[torch.Tensor],
    outputs_cached: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute similarity metrics between baseline and cached outputs.

    Returns exact match rate and token-level agreement.
    """
    exact_matches = 0
    total_tokens = 0
    matching_tokens = 0

    for baseline, cached in zip(outputs_baseline, outputs_cached):
        if torch.equal(baseline, cached):
            exact_matches += 1

        total_tokens += baseline.numel()
        matching_tokens += (baseline == cached).sum().item()

    return {
        "exact_match_rate": exact_matches / len(outputs_baseline),
        "token_agreement": matching_tokens / total_tokens,
    }


def run_benchmark(
    model: MaskedDiffusionTransformer,
    diffusion: MaskedDiffusion,
    device: str,
    batch_size: int = 1,
    seq_len: int = 128,
    steps: int = 64,
    num_runs: int = 5,
    cache_reloading_steps: List[int] = [1, 2, 4, 8],
    compare_outputs: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparing baseline vs dKV-Cache variants.
    """
    results = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "steps": steps,
            "num_runs": num_runs,
            "device": device,
        },
        "baseline": {},
        "dkv_cache": {},
    }

    # Note: dKV-Cache uses confidence-based unmasking, so baseline must too for fair comparison
    strategy = "confidence"

    # Baseline (no cache)
    print("\n" + "=" * 60)
    print("Baseline (no dKV-Cache, confidence strategy)")
    print("=" * 60)

    torch.manual_seed(seed)
    baseline_times = measure_sampling_time(
        model, diffusion, batch_size, seq_len, steps, device,
        use_dkv=False, num_runs=num_runs, strategy=strategy
    )

    results["baseline"]["timing"] = baseline_times
    print(f"  Time: {baseline_times['mean']*1000:.2f} +/- {baseline_times['std']*1000:.2f} ms")
    print(f"  Range: [{baseline_times['min']*1000:.2f}, {baseline_times['max']*1000:.2f}] ms")

    # Collect baseline outputs for comparison
    baseline_outputs = []
    if compare_outputs:
        torch.manual_seed(seed)
        for _ in range(3):
            out = diffusion.sample(
                model, batch_size, seq_len, steps=steps, device=device,
                strategy=strategy, use_dkv_cache=False
            )
            baseline_outputs.append(out.clone())

    # dKV-Cache variants
    for reload_step in cache_reloading_steps:
        print("\n" + "-" * 60)
        print(f"dKV-Cache (reload every {reload_step} steps)")
        print("-" * 60)

        torch.manual_seed(seed)
        cache_times = measure_sampling_time(
            model, diffusion, batch_size, seq_len, steps, device,
            use_dkv=True, cache_reloading_step=reload_step, num_runs=num_runs,
            strategy=strategy
        )

        speedup = baseline_times["mean"] / cache_times["mean"]

        result_key = f"reload_{reload_step}"
        results["dkv_cache"][result_key] = {
            "timing": cache_times,
            "speedup": speedup,
        }

        print(f"  Time: {cache_times['mean']*1000:.2f} +/- {cache_times['std']*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Compare outputs
        if compare_outputs and baseline_outputs:
            torch.manual_seed(seed)
            cache_outputs = []
            for _ in range(3):
                out = diffusion.sample(
                    model, batch_size, seq_len, steps=steps, device=device,
                    strategy=strategy,
                    use_dkv_cache=True, cache_reloading_step=reload_step
                )
                cache_outputs.append(out.clone())

            similarity = compute_output_similarity(baseline_outputs, cache_outputs)
            results["dkv_cache"][result_key]["similarity"] = similarity
            print(f"  Exact match: {similarity['exact_match_rate']*100:.1f}%")
            print(f"  Token agreement: {similarity['token_agreement']*100:.1f}%")

    return results


def log_to_wandb(results: Dict[str, Any], cfg: DictConfig, model_cfg: ModelConfig):
    """Log benchmark results to wandb."""
    import wandb

    config = results["config"]

    # Initialize wandb
    run_name = cfg.get("tag", None) or f"benchmark_dkv_{config['seq_len']}_{config['steps']}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config={
            "benchmark": config,
            "model": model_cfg.__dict__,
        },
        tags=["benchmark", "dkv-cache"],
        reinit=True,
    )

    # Log baseline
    baseline_time = results["baseline"]["timing"]["mean"] * 1000
    wandb.log({
        "baseline/time_ms": baseline_time,
        "baseline/time_std_ms": results["baseline"]["timing"]["std"] * 1000,
    })

    # Log dKV-Cache variants
    table_data = [["Baseline (no cache)", baseline_time, 1.0, None, None]]

    for key, data in results["dkv_cache"].items():
        reload_step = int(key.split("_")[1])
        time_ms = data["timing"]["mean"] * 1000
        speedup = data["speedup"]

        log_dict = {
            f"dkv_cache/reload_{reload_step}/time_ms": time_ms,
            f"dkv_cache/reload_{reload_step}/time_std_ms": data["timing"]["std"] * 1000,
            f"dkv_cache/reload_{reload_step}/speedup": speedup,
        }

        exact_match = None
        token_agreement = None
        if "similarity" in data:
            exact_match = data["similarity"]["exact_match_rate"]
            token_agreement = data["similarity"]["token_agreement"]
            log_dict[f"dkv_cache/reload_{reload_step}/exact_match"] = exact_match
            log_dict[f"dkv_cache/reload_{reload_step}/token_agreement"] = token_agreement

        wandb.log(log_dict)
        table_data.append([f"dKV-Cache (reload={reload_step})", time_ms, speedup, exact_match, token_agreement])

    # Create summary table
    table = wandb.Table(
        columns=["Method", "Time (ms)", "Speedup", "Exact Match", "Token Agreement"],
        data=table_data
    )
    wandb.log({"benchmark_summary": table})

    # Create speedup bar chart
    methods = ["Baseline"] + [f"reload={k.split('_')[1]}" for k in results["dkv_cache"].keys()]
    speedups = [1.0] + [d["speedup"] for d in results["dkv_cache"].values()]
    wandb.log({
        "speedup_chart": wandb.plot.bar(
            wandb.Table(data=[[m, s] for m, s in zip(methods, speedups)], columns=["Method", "Speedup"]),
            "Method", "Speedup", title="dKV-Cache Speedup Comparison"
        )
    })

    wandb.finish()
    print("\nResults logged to wandb.")


def print_summary(results: Dict[str, Any], dataset=None):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    config = results["config"]
    print(f"\nConfig: batch_size={config['batch_size']}, seq_len={config['seq_len']}, "
          f"steps={config['steps']}, device={config['device']}")

    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Method", "Time (ms)", "Speedup", "Exact Match", "Token Agr."
    ))
    print("-" * 70)

    baseline_time = results["baseline"]["timing"]["mean"] * 1000
    print("{:<25} {:>12.2f} {:>12} {:>12} {:>12}".format(
        "Baseline (no cache)", baseline_time, "1.00x", "-", "-"
    ))

    for key, data in results["dkv_cache"].items():
        reload_step = key.split("_")[1]
        time_ms = data["timing"]["mean"] * 1000
        speedup = data["speedup"]

        if "similarity" in data:
            exact = f"{data['similarity']['exact_match_rate']*100:.1f}%"
            token = f"{data['similarity']['token_agreement']*100:.1f}%"
        else:
            exact, token = "-", "-"

        print("{:<25} {:>12.2f} {:>12.2f}x {:>12} {:>12}".format(
            f"dKV-Cache (reload={reload_step})", time_ms, speedup, exact, token
        ))

    print("-" * 70)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    best_speedup = 0
    best_key = None
    for key, data in results["dkv_cache"].items():
        if "similarity" in data and data["similarity"]["token_agreement"] > 0.95:
            if data["speedup"] > best_speedup:
                best_speedup = data["speedup"]
                best_key = key

    if best_key:
        reload = best_key.split("_")[1]
        print(f"  Best setting with >95% token agreement: cache_reloading_step={reload} ({best_speedup:.2f}x speedup)")
    else:
        print("  Use cache_reloading_step=1 for maximum quality (no caching benefit)")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Get checkpoint path
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        print("Usage: python benchmark_dkv.py checkpoint=path/to/ckpt.pt")
        print("\nOptions:")
        print("  num_runs=5              Number of timed runs per config")
        print("  batch_size=1            Batch size for generation")
        print("  seq_len=128             Sequence length to generate")
        print("  steps=64                Number of diffusion steps")
        print("  cache_reloading_steps=[1,2,4,8]  Cache reload intervals to test")
        print("  compare_outputs=true    Compare output quality")
        return

    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint = os.path.join(orig_cwd, checkpoint)

    # Setup device
    device = cfg.system.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    # Benchmark parameters
    num_runs = cfg.get("num_runs", 5)
    batch_size = cfg.get("batch_size", 1)
    seq_len = cfg.get("seq_len", 128)
    steps = cfg.get("steps", cfg.sampling.steps)
    compare_outputs = cfg.get("compare_outputs", True)

    # Parse cache_reloading_steps (can be list or single value)
    cache_steps = cfg.get("cache_reloading_steps", [1, 2, 4, 8])
    if isinstance(cache_steps, int):
        cache_steps = [cache_steps]
    elif isinstance(cache_steps, str):
        cache_steps = [int(x) for x in cache_steps.strip("[]").split(",")]
    else:
        cache_steps = list(cache_steps)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["config"])
    model = MaskedDiffusionTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Get diffusion eps
    eps = 1e-3
    if "hydra_cfg" in ckpt:
        eps = ckpt["hydra_cfg"].get("diffusion", {}).get("eps", 1e-3)
    diffusion = MaskedDiffusion(model_cfg.vocab_size, eps=eps)

    # Load dataset for decoding (optional)
    data_dir = os.path.join(orig_cwd, cfg.data.dir)
    try:
        dataset = TextDataset(data_dir)
    except:
        dataset = None

    # Print setup info
    print(f"\nModel: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run benchmark
    results = run_benchmark(
        model, diffusion, device,
        batch_size=batch_size,
        seq_len=seq_len,
        steps=steps,
        num_runs=num_runs,
        cache_reloading_steps=cache_steps,
        compare_outputs=compare_outputs,
        seed=cfg.system.seed,
    )

    # Print summary
    print_summary(results, dataset)

    # Log to wandb if enabled
    if cfg.wandb.enabled:
        log_to_wandb(results, cfg, model_cfg)

    # Show sample outputs
    if dataset:
        print("\n" + "=" * 70)
        print("SAMPLE OUTPUTS")
        print("=" * 70)

        torch.manual_seed(cfg.system.seed)

        print("\nBaseline (no cache):")
        out = diffusion.sample(model, 1, min(seq_len, 100), steps=steps, device=device,
                              strategy="confidence", use_dkv_cache=False)
        ids = [t for t in out[0].tolist() if t < model_cfg.vocab_size]
        print(f"  {dataset.decode(ids)[:200]}...")

        best_reload = cache_steps[-1] if len(cache_steps) > 1 else cache_steps[0]
        print(f"\ndKV-Cache (reload={best_reload}):")
        torch.manual_seed(cfg.system.seed)
        out = diffusion.sample(model, 1, min(seq_len, 100), steps=steps, device=device,
                              strategy="confidence",
                              use_dkv_cache=True, cache_reloading_step=best_reload)
        ids = [t for t in out[0].tolist() if t < model_cfg.vocab_size]
        print(f"  {dataset.decode(ids)[:200]}...")


if __name__ == "__main__":
    main()
