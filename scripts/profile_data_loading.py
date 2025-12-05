#!/usr/bin/env python3
"""
Profile data loading to identify bottlenecks in the training pipeline.

This script measures:
1. Time to load a single sample from esp-data (including GCS download)
2. Time for the collate function (padding, windowing)
3. Time for a full batch through the DataLoader
4. Model forward pass time (for comparison)

Usage:
    # On Slurm (interactive session):
    srun --partition=a100-40 --gpus=1 --time=0:30:00 --pty bash
    cd ~/representation-learning
    uv run python scripts/profile_data_loading.py --config configs/run_configs/pretrained/openbeats_ft.yml

    # Or submit as a job:
    sbatch jobs/profile_data_loading.sh
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def profile_single_sample(dataset, num_samples: int = 10) -> dict[str, float]:
    """Profile loading individual samples from the dataset."""
    times = []

    logger.info(f"Profiling {num_samples} individual sample loads...")

    for i in range(min(num_samples, len(dataset))):
        start = time.perf_counter()
        sample = dataset[i]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if i == 0:
            # Log first sample details
            audio_key = "audio" if "audio" in sample else "raw_wav"
            if audio_key in sample:
                audio = sample[audio_key]
                logger.info(
                    f"  Sample 0: audio shape={audio.shape if hasattr(audio, 'shape') else len(audio)}, "
                    f"load time={elapsed:.3f}s"
                )

    return {
        "mean_sample_time": sum(times) / len(times),
        "min_sample_time": min(times),
        "max_sample_time": max(times),
        "first_sample_time": times[0] if times else 0,
    }


def profile_dataloader(dataloader, num_batches: int = 10) -> dict[str, float]:
    """Profile loading batches through the DataLoader."""
    times = []
    batch_sizes = []

    logger.info(f"Profiling {num_batches} batch loads through DataLoader...")

    start_total = time.perf_counter()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        elapsed = time.perf_counter() - start_total
        times.append(elapsed)

        audio_key = "raw_wav" if "raw_wav" in batch else "audio"
        if audio_key in batch:
            batch_sizes.append(batch[audio_key].shape[0])

        if i == 0:
            logger.info(
                f"  Batch 0: shape={batch[audio_key].shape if audio_key in batch else 'N/A'}, "
                f"time={elapsed:.3f}s"
            )

        start_total = time.perf_counter()

    # Calculate inter-batch times
    inter_batch_times = times[1:] if len(times) > 1 else times

    return {
        "mean_batch_time": sum(inter_batch_times) / len(inter_batch_times)
        if inter_batch_times
        else 0,
        "first_batch_time": times[0] if times else 0,
        "batches_per_second": len(inter_batch_times) / sum(inter_batch_times)
        if sum(inter_batch_times) > 0
        else 0,
        "samples_per_second": sum(batch_sizes) / sum(times) if sum(times) > 0 else 0,
    }


def profile_model_forward(
    model, batch: dict, device: str, num_iterations: int = 10
) -> dict[str, float]:
    """Profile model forward pass time."""
    times = []

    logger.info(f"Profiling {num_iterations} model forward passes...")

    # Move batch to device
    batch_gpu = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Warmup
    with torch.no_grad():
        audio_key = "raw_wav" if "raw_wav" in batch_gpu else "audio"
        if audio_key in batch_gpu:
            _ = model(batch_gpu[audio_key])

    if device == "cuda":
        torch.cuda.synchronize()

    for i in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            if audio_key in batch_gpu:
                _ = model(batch_gpu[audio_key])
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_forward_time": sum(times) / len(times),
        "min_forward_time": min(times),
        "max_forward_time": max(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile data loading pipeline")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to run config YAML"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to profile"
    )
    parser.add_argument(
        "--num-batches", type=int, default=10, help="Number of batches to profile"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for model profiling",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DATA LOADING PROFILER")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {args.device}")

    # Import here to avoid slow startup for --help
    from representation_learning.configs import RunConfig
    from representation_learning.data.dataset import build_dataloaders
    from representation_learning.models.get_model import get_model

    # Load config
    config = RunConfig.from_yaml(args.config)

    logger.info(f"\nDataset: {config.dataset_config}")
    logger.info(f"Model: {config.model_spec.name}")
    logger.info(f"Batch size: {config.training_params.batch_size}")
    logger.info(f"Num workers: {config.num_workers}")

    # Build dataloaders
    logger.info("\n" + "=" * 60)
    logger.info("BUILDING DATALOADERS...")
    logger.info("=" * 60)

    start = time.perf_counter()
    train_dl, val_dl, _ = build_dataloaders(config, device=args.device)
    dataloader_build_time = time.perf_counter() - start
    logger.info(f"DataLoader build time: {dataloader_build_time:.2f}s")

    # Profile individual samples
    logger.info("\n" + "=" * 60)
    logger.info("PROFILING INDIVIDUAL SAMPLES (esp-data + GCS)")
    logger.info("=" * 60)

    sample_stats = profile_single_sample(train_dl.dataset, args.num_samples)
    logger.info(f"  Mean sample load time: {sample_stats['mean_sample_time']:.3f}s")
    logger.info(f"  First sample (cold): {sample_stats['first_sample_time']:.3f}s")
    logger.info(
        f"  Min/Max: {sample_stats['min_sample_time']:.3f}s / {sample_stats['max_sample_time']:.3f}s"
    )

    # Profile dataloader batches
    logger.info("\n" + "=" * 60)
    logger.info("PROFILING DATALOADER BATCHES")
    logger.info("=" * 60)

    batch_stats = profile_dataloader(train_dl, args.num_batches)
    logger.info(f"  Mean batch load time: {batch_stats['mean_batch_time']:.3f}s")
    logger.info(f"  First batch (cold): {batch_stats['first_batch_time']:.3f}s")
    logger.info(f"  Batches/second: {batch_stats['batches_per_second']:.2f}")
    logger.info(f"  Samples/second: {batch_stats['samples_per_second']:.2f}")

    # Profile model forward pass
    logger.info("\n" + "=" * 60)
    logger.info("PROFILING MODEL FORWARD PASS")
    logger.info("=" * 60)

    # Get one batch for model profiling
    batch = next(iter(train_dl))

    # Load model
    model = get_model(
        config.model_spec,
        num_classes=train_dl.dataset.metadata.get("num_classes", 1000),
    )
    model = model.to(args.device)
    model.eval()

    forward_stats = profile_model_forward(model, batch, args.device)
    logger.info(f"  Mean forward time: {forward_stats['mean_forward_time']:.3f}s")
    logger.info(
        f"  Min/Max: {forward_stats['min_forward_time']:.3f}s / {forward_stats['max_forward_time']:.3f}s"
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY & BOTTLENECK ANALYSIS")
    logger.info("=" * 60)

    batch_size = config.training_params.batch_size
    theoretical_batch_time = sample_stats["mean_sample_time"] * batch_size

    logger.info(f"\nWith {config.num_workers} DataLoader workers:")
    logger.info(f"  Theoretical batch time (serial): {theoretical_batch_time:.3f}s")
    logger.info(f"  Actual batch time: {batch_stats['mean_batch_time']:.3f}s")
    logger.info(f"  Model forward time: {forward_stats['mean_forward_time']:.3f}s")

    if batch_stats["mean_batch_time"] > forward_stats["mean_forward_time"] * 2:
        logger.warning("\n⚠️  DATA LOADING IS THE BOTTLENECK!")
        logger.warning("   The GPU is waiting for data most of the time.")
        logger.warning("\n   Possible solutions:")
        logger.warning("   1. Increase num_workers (currently: %d)", config.num_workers)
        logger.warning("   2. Use local storage instead of GCS bucket")
        logger.warning("   3. Pre-cache data to local scratch before training")
        logger.warning("   4. Use a faster storage backend (NVMe SSD)")
    else:
        logger.info("\n✓ Data loading is NOT the bottleneck.")
        logger.info("  Model forward pass is the limiting factor (as expected).")

    # Estimate training time
    steps_per_epoch = len(train_dl)
    time_per_step = batch_stats["mean_batch_time"] + forward_stats["mean_forward_time"]
    epoch_time_estimate = steps_per_epoch * time_per_step

    logger.info(f"\nEstimated epoch time: {epoch_time_estimate / 60:.1f} minutes")
    logger.info(f"  ({steps_per_epoch} steps × {time_per_step:.2f}s/step)")


if __name__ == "__main__":
    main()
