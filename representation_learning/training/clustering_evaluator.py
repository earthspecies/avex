"""
Clustering evaluation component for training.

This module provides clustering evaluation functionality that can be integrated
into the training loop to monitor unsupervised clustering quality.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.configs import ClusteringEvalConfig
from representation_learning.training.distributed import is_main_process

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """Handles clustering evaluation during training."""

    def __init__(self, config: ClusteringEvalConfig, device: torch.device) -> None:
        """Initialize the clustering evaluator.

        Parameters
        ----------
        config : ClusteringEvalConfig
            Configuration for clustering evaluation
        device : torch.device
            Device for computation
        """
        self.config = config
        self.device = device
        self.layer_names = self._parse_layer_names(config.layers)

    def _get_aggregation_method(self, model: torch.nn.Module) -> str:
        """Determine the appropriate aggregation method based on model type.

        Parameters
        ----------
        model : torch.nn.Module
            The model to determine aggregation for

        Returns
        -------
        str
            Aggregation method: 'cls_token' for transformers, 'mean' for CNNs
        """
        # Transformer-based models that support cls_token
        # if any(transformer_name in model_class_name for transformer_name in
        # transformer_models):
        #     return "cls_token"
        # else:
        #     # CNN-based models (EfficientNet, etc.)
        #     return "mean"
        return "mean"

    def _parse_layer_names(self, layers_str: str) -> List[str]:
        """Parse layer names from configuration string.

        Parameters
        ----------
        layers_str : str
            Comma-separated layer names or 'last_layer'

        Returns
        -------
        List[str]
            List of layer names
        """
        if layers_str.strip() == "last_layer":
            return ["last_layer"]
        return [name.strip() for name in layers_str.split(",")]

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Extract embeddings and compute clustering metrics.

        Parameters
        ----------
        model : torch.nn.Module
            Model to extract embeddings from
        dataloader : DataLoader
            Dataloader to extract embeddings from
        epoch : int
            Current epoch number

        Returns
        -------
        Dict[str, float]
            Dictionary of clustering metrics
        """
        if not is_main_process():
            return {}

        logger.info(f"Running clustering evaluation at epoch {epoch}")

        try:
            # Extract embeddings from validation set
            embeddings, labels = self._extract_embeddings(model, dataloader)

            if embeddings.numel() == 0 or labels.numel() == 0:
                logger.warning("No embeddings extracted for clustering evaluation")
                return {}

            # Compute clustering metrics
            from representation_learning.evaluation.clustering import (
                eval_clustering,
                eval_clustering_multiple_k,
            )

            metrics = {}
            metrics.update(eval_clustering(embeddings, labels))
            metrics.update(eval_clustering_multiple_k(embeddings, labels))

            logger.info(f"Clustering evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Clustering evaluation failed: {e}")
            return {}

    def _extract_embeddings(
        self, model: torch.nn.Module, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings from dataloader.

        Parameters
        ----------
        model : torch.nn.Module
            Model to extract embeddings from
        dataloader : DataLoader
            Dataloader to process

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of (embeddings, labels) on CPU
        """
        model.eval()
        embeddings_list = []
        labels_list = []
        sample_count = 0

        # Handle layer name resolution for models that need it
        layer_names = self.layer_names.copy()
        if "last_layer" in layer_names:
            # Find the actual last linear layer name
            linear_layers = [
                n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
            ]
            if linear_layers:
                # Replace 'last_layer' with actual layer name
                layer_names = [
                    linear_layers[-1] if name == "last_layer" else name
                    for name in layer_names
                ]
            else:
                logger.warning("No linear layers found, using 'last_layer' as-is")

        with torch.no_grad():
            # Register hooks for the specified layers outside the loop
            if hasattr(model, "register_hooks_for_layers"):
                model.register_hooks_for_layers(layer_names)
            else:
                logger.warning("Model does not support register_hooks_for_layers")

            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                if self.config.max_samples and sample_count >= self.config.max_samples:
                    break

                # Move batch to device
                wav = batch["raw_wav"].to(self.device)
                labels = batch["label"]
                padding_mask = batch.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(self.device)

                # Extract embeddings
                try:
                    # Determine aggregation method based on model type
                    aggregation_method = self._get_aggregation_method(model)

                    if padding_mask is None:
                        embeddings = model.extract_embeddings(
                            wav, aggregation=aggregation_method
                        )
                    else:
                        inp = {"raw_wav": wav, "padding_mask": padding_mask}
                        embeddings = model.extract_embeddings(
                            inp, aggregation=aggregation_method
                        )

                    # Move to CPU for memory efficiency
                    embeddings_list.append(embeddings.cpu())
                    labels_list.append(labels.cpu())

                    sample_count += len(labels)

                    # Limit samples if configured
                    if (
                        self.config.max_samples
                        and sample_count >= self.config.max_samples
                    ):
                        # Truncate last batch if needed
                        excess = sample_count - self.config.max_samples
                        if excess > 0:
                            embeddings_list[-1] = embeddings_list[-1][:-excess]
                            labels_list[-1] = labels_list[-1][:-excess]
                        break

                except Exception as e:
                    logger.warning(f"Failed to extract embeddings for batch: {e}")
                    continue

        if hasattr(model, "deregister_all_hooks"):
            model.deregister_all_hooks()
        else:
            logger.warning("Model does not support deregister_all_hooks")

        if not embeddings_list:
            logger.warning("No embeddings were successfully extracted")
            return torch.empty(0), torch.empty(0)

        logger.info(f"Extracted embeddings from {sample_count} samples")
        all_embeddings = torch.cat(embeddings_list)
        all_labels = torch.cat(labels_list)

        # Handle one-hot encoded labels
        if all_labels.dim() > 1:
            # Convert one-hot to class indices
            all_labels = torch.argmax(all_labels, dim=1)

        # Filter out classes with only a single instance
        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)
        multi_instance_mask = label_counts > 1
        valid_labels = unique_labels[multi_instance_mask]

        if len(valid_labels) == 0:
            logger.warning(
                "No classes with multiple instances found for clustering evaluation"
            )
            return torch.empty(0), torch.empty(0)

        # Create mask for samples belonging to classes with multiple instances
        sample_mask = torch.isin(all_labels, valid_labels)
        filtered_embeddings = all_embeddings[sample_mask]
        filtered_labels = all_labels[sample_mask]

        num_filtered = len(all_labels) - len(filtered_labels)
        if num_filtered > 0:
            logger.info(
                f"Filtered out {num_filtered} samples from "
                f"{len(unique_labels) - len(valid_labels)} single-instance classes"
            )

        logger.info(
            f"Using {len(filtered_labels)} samples from {len(valid_labels)} classes "
            f"for clustering evaluation"
        )
        return filtered_embeddings, filtered_labels
