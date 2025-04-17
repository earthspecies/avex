"""
Entry-point script for linear probing/fine-tuning experiments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import h5py

from representation_learning.configs import load_config, RunConfig, EvaluateConfig  # type: ignore
from representation_learning.models.get_model import get_model
from representation_learning.models.linear_probe import LinearProbe
from representation_learning.data.dataset import build_evaluation_dataloaders
from representation_learning.training.train import Trainer
from representation_learning.training.optimisers import get_optimizer
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger("run_finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")

@dataclass
class ExperimentResult:
    dataset_name: str
    experiment_name: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe/fine-tune an audio representation model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the evaluation config YAML (see configs/evaluation_configs/*)"
    )
    return parser.parse_args()

def get_embeddings(model: torch.nn.Module, x: torch.Tensor, layers: List[str]) -> torch.Tensor:
    """
    Extract embeddings from specified layers of the model.
    
    Args:
        model: The model to extract embeddings from
        x: Input tensor
        layers: List of layer names to extract embeddings from
        
    Returns:
        Concatenated embeddings from specified layers
    """
    embeddings = []
    
    def hook_fn(module, input, output):
        embeddings.append(output)
    
    hooks = []
    for name, module in model.named_modules():
        if name in layers:
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate embeddings
    if not embeddings:
        raise ValueError(f"No layers found matching: {layers}")
    
    return torch.cat([e.flatten(start_dim=1) for e in embeddings], dim=1)

def save_embeddings_to_disk(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                          layer_names: List[str], save_dir: Path, split: str) -> None:
    """
    Save embeddings for all samples in a dataloader to disk using HDF5 format.
    
    Args:
        model: The model to extract embeddings from
        dataloader: DataLoader containing the samples
        layer_names: List of layer names to extract embeddings from
        save_dir: Directory to save embeddings
        split: Dataset split name (e.g., 'train' or 'val')
    """
    save_dir = save_dir / "embeddings"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file for this split
    h5_path = save_dir / f"{split}.h5"
    model.eval()
    
    # Log system information
    logger.info("Starting embedding save process")
    logger.info(f"Save path: {h5_path}")
    logger.info(f"Total samples: {len(dataloader.dataset)}")
    logger.info(f"Batch size: {dataloader.batch_size}")
    logger.info(f"Number of workers: {dataloader.num_workers}")
    
    # Get system limits
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"File descriptor limits - soft: {soft}, hard: {hard}")
    except Exception as e:
        logger.warning(f"Could not get file descriptor limits: {e}")
    
    try:
        with h5py.File(h5_path, 'w', libver='latest') as h5f:
            # Get first batch to determine shapes
            first_batch = next(iter(dataloader))
            sample_embeddings = get_embeddings(model, first_batch[0], layer_names)
            embedding_dim = sample_embeddings.shape[1]
            total_samples = len(dataloader.dataset)
            
            # Calculate optimal chunk size based on available memory
            available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8 * 1024 * 1024 * 1024  # 8GB default
            chunk_size = min(100, max(1, int(available_memory / (embedding_dim * 4 * 1024 * 1024))))  # 4 bytes per float32
            
            logger.info(f"Embedding dimension: {embedding_dim}")
            logger.info(f"Chunk size: {chunk_size}")
            
            # Create resizable datasets with compression
            embeddings_dset = h5f.create_dataset(
                'embeddings',
                shape=(total_samples, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
                chunks=(chunk_size, embedding_dim),
                compression='gzip',
                compression_opts=4
            )
            labels_dset = h5f.create_dataset(
                'labels',
                shape=(total_samples,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(chunk_size,),
                compression='gzip',
                compression_opts=4
            )
            
            # Save embeddings and labels in chunks
            start_idx = 0
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"Saving {split} embeddings")):
                    try:
                        # Log progress every 10 batches
                        if batch_idx % 10 == 0:
                            logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
                        
                        embeddings = get_embeddings(model, x, layer_names)
                        batch_size = len(embeddings)
                        
                        # Write to HDF5
                        embeddings_dset[start_idx:start_idx + batch_size] = embeddings.cpu().numpy()
                        labels_dset[start_idx:start_idx + batch_size] = y.numpy()
                        
                        start_idx += batch_size
                        
                        # Clear memory
                        del embeddings
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Force flush to disk periodically
                        if batch_idx % 100 == 0:
                            h5f.flush()
                            
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx} at index {start_idx}: {str(e)}")
                        raise
                        
    except Exception as e:
        logger.error(f"Error saving embeddings to {h5_path}: {str(e)}")
        if h5_path.exists():
            h5_path.unlink()  # Remove partial file
        raise

def load_embeddings_from_disk(save_dir: Path, split: str) -> torch.utils.data.Dataset:
    """
    Create a dataset from saved embeddings in HDF5 format.
    
    Args:
        save_dir: Directory containing saved embeddings
        split: Dataset split name (e.g., 'train' or 'val')
        
    Returns:
        Dataset containing embeddings and labels
    """
    class HDF5EmbeddingDataset(torch.utils.data.Dataset):
        def __init__(self, h5_path: Path):
            self.h5_path = h5_path
            self.h5_file = h5py.File(h5_path, 'r')
            self.embeddings = self.h5_file['embeddings']
            self.labels = self.h5_file['labels']
            
        def __len__(self):
            return len(self.embeddings)
            
        def __getitem__(self, idx):
            # HDF5 handles efficient reading of individual samples
            return torch.from_numpy(self.embeddings[idx]), torch.from_numpy(self.labels[idx])
        
        def __del__(self):
            # Ensure HDF5 file is closed when dataset is deleted
            if hasattr(self, 'h5_file'):
                self.h5_file.close()
    
    return HDF5EmbeddingDataset(save_dir / "embeddings" / f"{split}.h5")

def run_experiment(
    eval_cfg: EvaluateConfig,
    dataset_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    device: torch.device,
    save_dir: Path
) -> ExperimentResult:
    """
    Run a single experiment on a single dataset.
    
    Args:
        dataset_config: Configuration for the dataset
        experiment_config: Configuration for the experiment
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Experiment results
    """
    dataset_name = dataset_config.name
    experiment_name = experiment_config.run_name
    
    logger.info("Running experiment '%s' on dataset '%s'", experiment_name, dataset_name)

    # 1. Load run config for the experiment
    original_run_cfg: RunConfig = load_config(experiment_config.run_config)

    # 2. Build the dataloaders
    train_dl, val_dl = build_evaluation_dataloaders(eval_cfg, original_run_cfg.model_spec, dataset_config, device=device)
    logger.info(
        "Dataset ready: %d training batches / %d validation batches",
        len(train_dl), len(val_dl)
    )
    
    # 3. Get number of classes
    num_labels = len(train_dl.dataset.label2idx)
    logger.info("Number of labels: %d", num_labels)

    # 4. Load the pretrained model
    original_run_cfg.model_spec.pretrained = experiment_config.pretrained
    model = get_model(original_run_cfg.model_spec, num_classes=num_labels).to(device)
    
    # If pretrained=True, we don't need to load a checkpoint
    if not experiment_config.pretrained:
        # Load the latest checkpoint
        ckpt_path = Path("checkpoints") / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
        logger.info("Loaded model checkpoint from %s", ckpt_path)
    
    model.eval()
    logger.info("Model → %s parameters", sum(p.numel() for p in model.parameters()))

    # 5. Get layer names for embedding extraction
    if experiment_config.layers == "last_layer":
        layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        layer_names = [layer_names[-1]]
    else:
        layer_names = experiment_config.layers.split(",")
    logger.info("Layers: %s", layer_names)
    
    # 6. Save embeddings to disk
    logger.info("Saving embeddings to disk...")
    save_embeddings_to_disk(model, train_dl, layer_names, save_dir, "train")
    save_embeddings_to_disk(model, val_dl, layer_names, save_dir, "val")
    
    # 7. Create datasets from saved embeddings
    train_embedding_ds = load_embeddings_from_disk(save_dir, "train")
    val_embedding_ds = load_embeddings_from_disk(save_dir, "val")
    
    # 8. Create new dataloaders for embeddings with appropriate batch size
    train_embedding_dl = torch.utils.data.DataLoader(
        train_embedding_ds,
        batch_size=eval_cfg.training_params.batch_size,
        shuffle=True,
        num_workers=eval_cfg.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(eval_cfg.num_workers > 0)  # Keep workers alive between epochs
    )
    val_embedding_dl = torch.utils.data.DataLoader(
        val_embedding_ds,
        batch_size=eval_cfg.training_params.batch_size,
        shuffle=False,
        num_workers=eval_cfg.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(eval_cfg.num_workers > 0)  # Keep workers alive between epochs
    )
    
    # 9. Get embedding dimension from first sample
    sample_embedding, _ = train_embedding_ds[0]
    embedding_dim = sample_embedding.shape[0]
    logger.info("Embedding dimension: %d", embedding_dim)
    
    # 10. Create linear probe
    linear_probe = LinearProbe(embedding_dim, num_labels, device=device)
    logger.info("Linear probe → %s parameters", sum(p.numel() for p in linear_probe.parameters()))

    # 11. Create optimizer and trainer
    optim = get_optimizer(linear_probe.parameters(), experiment_config.training_params)
    
    # Create experiment-specific logger
    exp_logger = ExperimentLogger.from_config(experiment_config)
    exp_logger.log_dir = save_dir / dataset_name / experiment_name
    exp_logger.log_dir.mkdir(parents=True, exist_ok=True)

    # 12. Create a custom trainer for linear probing
    class LinearProbeTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Override the criterion based on multi_label flag
            self.multi_label = dataset_config.get("multi_label", False)
            if self.multi_label:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

        def _forward(self, batch, train: bool):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward through linear probe
            logits = linear_probe(x)
            loss = self.criterion(logits, y)
            
            # Calculate accuracy based on multi_label flag
            if self.multi_label:
                # For multi-label, use threshold of 0.5 and calculate accuracy per sample
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct = (pred == y).all(dim=1).sum().item()
            else:
                # For single-label, use argmax
                pred = logits.argmax(dim=1)
                correct = (pred == y).sum().item()
            
            return loss.item(), correct, y.size(0)

    trainer = LinearProbeTrainer(
        model=linear_probe,  # Now we use the linear probe directly
        optimizer=optim,
        train_loader=train_embedding_dl,
        val_loader=val_embedding_dl,
        device=device,
        cfg=experiment_config,
        exp_logger=exp_logger
    )

    # 13. Train the linear probe
    trainer.train(num_epochs=experiment_config.training_params.train_epochs)
    
    # 14. Get final metrics
    train_metrics = trainer._run_epoch(train=True, epoch=experiment_config.training_params.train_epochs)
    val_metrics = trainer._run_epoch(train=False, epoch=experiment_config.training_params.train_epochs)
    
    return ExperimentResult(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        train_metrics={"loss": train_metrics[0], "acc": train_metrics[1]},
        val_metrics={"loss": val_metrics[0], "acc": val_metrics[1]}
    )

def main() -> None:
    args = _parse_args()

    # 1. Load evaluation config
    eval_cfg: EvaluateConfig = load_config(args.config, config_type="evaluate")
    logger.info("Loaded evaluation config from %s", args.config)
    
    # 2. Load dataset config
    dataset_cfg = load_config(eval_cfg.dataset_config, config_type="benchmark")
    logger.info("Loaded benchmark dataset config from %s", eval_cfg.dataset_config)
    
    # 3. Create save directory
    save_dir = Path(eval_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)  # Fixed seed for reproducibility
    
    # 5. Run experiments for each dataset and experiment combination
    for dataset in dataset_cfg.datasets:
        results = []
        for experiment in eval_cfg.experiments:
            result = run_experiment(eval_cfg, dataset, experiment, device, save_dir)
            results.append(result)
            
            # Log results
            logger.info(
                "Results for dataset '%s', experiment '%s':\n"
                "  Train: loss=%.4f, acc=%.4f\n"
                "  Val:   loss=%.4f, acc=%.4f",
                result.dataset_name,
                result.experiment_name,
                result.train_metrics["loss"],
                result.train_metrics["acc"],
                result.val_metrics["loss"],
                result.val_metrics["acc"]
            )
    

    # 6. Save summary of all results
    summary_path = save_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Experiment Summary\n")
        f.write("================\n\n")
        for result in results:
            f.write(f"Dataset: {result.dataset_name}\n")
            f.write(f"Experiment: {result.experiment_name}\n")
            f.write(f"Train metrics: {result.train_metrics}\n")
            f.write(f"Validation metrics: {result.val_metrics}\n")
            f.write("-" * 50 + "\n")
    
    logger.info("Saved summary to %s", summary_path)

if __name__ == "__main__":
    main() 