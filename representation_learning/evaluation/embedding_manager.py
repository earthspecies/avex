"""Embedding management utilities for offline evaluation.

This module provides utilities for managing embedding datasets, including
loading, saving, and caching embeddings for offline evaluation tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import h5py
import torch

from representation_learning.evaluation.embedding_utils import (
    EmbeddingDataset,
    HDF5EmbeddingDataset,
    _extract_embeddings_in_memory,
    _extract_embeddings_streaming,
    load_embeddings_arrays,
    save_embeddings_arrays,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingDataSourceConfig:
    """Configuration for embedding data source.

    Specifies the path where embeddings are saved and loaded from.
    """

    save_path: Path
    memory_limit_bytes: int = 32 * 1024 * 1024 * 1024  # 32GB default
    # If explicitly set, overrides automatic decisions for both load & compute paths
    use_streaming_embeddings: Optional[bool] = None
    cache_size_limit_gb: float = 8.0
    chunk_size: int = 1000
    compression: str = "gzip"
    compression_level: int = 4
    auto_chunk_size: bool = True
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    batch_chunk_size: int = 10
    disable_tqdm: bool = False
    disable_layerdrop: Optional[bool] = None


class EmbeddingDataSource:
    """Unified interface for embedding computation and dataset creation.

    - If `base_model` and `dataloader` are provided, computes embeddings
      (streaming or in-memory decided automatically) and saves/returns a dataset.
    - If they are not provided, loads an existing HDF5-backed dataset from `save_path`.
    - Returns a dataset compatible with finetuning: either `EmbeddingDataset`
      (in-memory) or `HDF5EmbeddingDataset` (on-disk lazy loading).
    """

    def __init__(
        self,
        *,
        save_path: Path,
        layer_names: List[str],
        aggregation: str = "mean",
        config: Optional[EmbeddingDataSourceConfig] = None,
    ) -> None:
        self.save_path = save_path
        self.layer_names = layer_names
        self.aggregation = aggregation
        self.config = config or EmbeddingDataSourceConfig(save_path=save_path)
        self.embedding_dims: Optional[List[Tuple[int, ...]]] = None
        self.num_labels: Optional[int] = None

    def _estimate_dataset_bytes(
        self, sample_shape: Tuple[int, ...], num_samples: int, dtype_size: int = 4
    ) -> int:
        # float32 is 4 bytes by default
        elements_per_sample = 1
        for dim in sample_shape:
            elements_per_sample *= int(dim)
        return elements_per_sample * dtype_size * num_samples

    def _should_stream(self, first_sample_shapes: List[Tuple[int, ...]], num_samples: int) -> bool:
        # Sum layer sizes to estimate total footprint per sample
        # Account for aggregation method that reduces dimensions
        total_per_sample = 0
        for shape in first_sample_shapes:
            prod = 1
            for d in shape:
                prod *= int(d)

            # Apply aggregation reduction
            if self.aggregation == "mean" or self.aggregation == "max":
                # Mean/max aggregation reduces sequence dimension
                # (typically 2nd dimension)
                if len(shape) >= 2:
                    # Remove the sequence dimension (assume it's the 2nd dimension)
                    prod = prod // shape[1] if len(shape) > 1 else prod
            elif self.aggregation == "none":
                # No aggregation, keep original shape
                pass
            # Add other aggregation methods as needed

            total_per_sample += prod

        est_bytes = total_per_sample * 4 * num_samples
        logger.info(
            f"Estimated embedding footprint: ~{est_bytes / 1e9:.2f} GB for "
            f"{num_samples} samples (aggregation: {self.aggregation})"
        )
        return est_bytes > self.config.memory_limit_bytes

    def get_dataset(
        self,
        *,
        base_model: Optional[torch.nn.Module] = None,
        dataloader: Optional[Iterable] = None,
        device: Optional[torch.device] = None,
    ) -> Union[EmbeddingDataset, HDF5EmbeddingDataset]:
        """Return a dataset of embeddings (in-memory or HDF5-backed).

        If `base_model` and `dataloader` are provided, computes embeddings first.
        Otherwise, attempts to load from `self.save_path`.

        Returns:
            Union[EmbeddingDataset, HDF5EmbeddingDataset]: The embedding dataset.

        Raises:
            ValueError: If required parameters are missing or file loading fails.
        """
        save_path = self.config.save_path

        # Check if file exists first
        file_exists = Path(save_path).exists()

        if not file_exists and (base_model is None or dataloader is None or device is None):
            raise ValueError(
                f"Embedding file {save_path} does not exist and cannot be computed "
                f"because base_model, dataloader, or device is None"
            )

        if file_exists and (base_model is None or dataloader is None or device is None):
            logger.info("Loading embeddings from existing file")
            # Decide in-memory vs HDF5-backed based on file size
            try:
                file_size = Path(save_path).stat().st_size
            except Exception:
                file_size = self.config.memory_limit_bytes + 1  # fallback to HDF5

            if (
                self.config.use_streaming_embeddings is True
                or file_size > self.config.memory_limit_bytes
            ):
                logger.info("Loading embeddings into HDF5-backed dataset")
                # HDF5-backed lazy loading; allow file to define layers when 'all'
                ds = HDF5EmbeddingDataset(
                    save_path,
                    layer_names=(None if ("all" in self.layer_names) else self.layer_names),
                    cache_in_memory=True,
                    cache_size_limit_gb=max(1.0, float(self.config.cache_size_limit_gb)),
                    allow_partial_cache=True,
                )
                self.embedding_dims = getattr(ds, "embedding_dims", None)
                # If 'all' was requested, adopt detected layer names from file
                if "all" in self.layer_names and hasattr(ds, "layer_names"):
                    self.layer_names = list(ds.layer_names)
                # If 'last_layer' was requested, resolve it to the actual
                # last layer name
                elif "last_layer" in self.layer_names and hasattr(ds, "layer_names"):
                    if ds.layer_names:
                        # Replace 'last_layer' with the actual last layer name
                        self.layer_names = [
                            name if name != "last_layer" else ds.layer_names[-1]
                            for name in self.layer_names
                        ]
                        logger.info(f"Resolved 'last_layer' to '{ds.layer_names[-1]}'")
                    else:
                        logger.warning("No layer names found in dataset to resolve 'last_layer'")
                # Read num_labels and embedding_dims from HDF5 if present
                try:
                    with h5py.File(str(save_path), "r") as h5f:
                        # Read num_labels
                        nl = h5f.attrs.get("num_labels", None)
                        if nl is not None and int(nl) > 0:
                            self.num_labels = int(nl)
                        else:
                            import numpy as _np

                            lbls = _np.asarray(h5f["labels"])  # type: ignore[index]
                            if lbls.ndim > 1 and lbls.shape[-1] > 1:
                                lbls = lbls.argmax(axis=-1)
                            self.num_labels = int(_np.unique(lbls).size)

                        # Read embedding_dims from H5 attributes if not already set
                        if self.embedding_dims is None:
                            embedding_dims_str = h5f.attrs.get("embedding_dims", None)
                            if embedding_dims_str is not None:
                                # Convert string representations back to tuples
                                self.embedding_dims = []
                                for dim_str in embedding_dims_str:
                                    # Parse string like "(100, 768)" back to tuple
                                    dim_str = dim_str.strip()
                                    if dim_str.startswith("(") and dim_str.endswith(")"):
                                        # Remove parentheses and split by comma
                                        inner = dim_str[1:-1].strip()
                                        if inner:
                                            # Split by comma and convert to int
                                            dims = [int(x.strip()) for x in inner.split(",")]
                                            self.embedding_dims.append(tuple(dims))
                                        else:
                                            # Empty tuple case
                                            self.embedding_dims.append(())
                                    else:
                                        # Single number case
                                        self.embedding_dims.append((int(dim_str),))
                                logger.info(
                                    f"Loaded embedding_dims from H5 attributes: "
                                    f"{self.embedding_dims}"
                                )
                except Exception as e:
                    raise ValueError("Failed to read metadata from HDF5 file") from e
                return ds
            else:
                logger.info("Loading embeddings into memory")
                # Load fully into memory
                embeds, labels, num_labels = load_embeddings_arrays(save_path)
                self.num_labels = int(num_labels) if num_labels is not None else None
                if isinstance(embeds, dict):
                    # If 'last_layer' was requested, resolve it to the actual
                    # last layer name
                    if "last_layer" in self.layer_names:
                        if embeds:
                            # Replace 'last_layer' with the actual last layer name
                            last_layer_name = list(embeds.keys())[-1]
                            self.layer_names = [
                                name if name != "last_layer" else last_layer_name
                                for name in self.layer_names
                            ]
                            logger.info(f"Resolved 'last_layer' to '{last_layer_name}'")
                        else:
                            logger.warning("No embeddings found to resolve 'last_layer'")

                    # If 'all' requested, use all layers; else filter to requested
                    if "all" in self.layer_names:
                        self.layer_names = list(embeds.keys())
                        self.embedding_dims = [tuple(t.shape[1:]) for t in embeds.values()]
                        return EmbeddingDataset(embeds, labels)
                    else:
                        filtered_embeds = {k: v for k, v in embeds.items() if k in self.layer_names}
                        self.embedding_dims = [tuple(t.shape[1:]) for t in filtered_embeds.values()]
                        return EmbeddingDataset(filtered_embeds, labels)
                else:
                    self.embedding_dims = [tuple(embeds.shape[1:])]
                    return EmbeddingDataset(embeds, labels)

        # Peek first batch to estimate size and decide strategy
        first_batch = next(iter(dataloader))
        base_model.eval()

        # Ensure hooks are registered consistently for the preview extraction
        original_disable_layerdrop = None
        if self.config.disable_layerdrop is not None and hasattr(base_model, "disable_layerdrop"):
            original_disable_layerdrop = base_model.disable_layerdrop
            base_model.disable_layerdrop = self.config.disable_layerdrop

        # Register hooks for requested layers
        if hasattr(base_model, "register_hooks_for_layers"):
            try:
                _ = base_model.register_hooks_for_layers(self.layer_names)
            except Exception:
                # Fall through; extract_embeddings may self-heal, but we try our best
                pass

        try:
            with torch.no_grad():
                wav = first_batch["raw_wav"].to(device)
                mask = first_batch.get("padding_mask")
                if mask is not None:
                    mask = mask.to(device)
                if mask is None:
                    sample_emb = base_model.extract_embeddings(wav, aggregation=self.aggregation)
                else:
                    sample_emb = base_model.extract_embeddings(
                        {"raw_wav": wav, "padding_mask": mask},
                        aggregation=self.aggregation,
                    )
        finally:
            # Always deregister hooks after preview to avoid duplicates later
            if hasattr(base_model, "deregister_all_hooks"):
                try:
                    base_model.deregister_all_hooks()
                except Exception:
                    pass
            if original_disable_layerdrop is not None and hasattr(base_model, "disable_layerdrop"):
                base_model.disable_layerdrop = original_disable_layerdrop

        # Resolve 'all' layer names from preview if requested
        if "all" in self.layer_names:
            if isinstance(sample_emb, dict):
                self.layer_names = sorted(list(sample_emb.keys()))
            elif isinstance(sample_emb, list):
                # Don't create synthetic names - let the model handle 'all'
                # resolution. Keep 'all' in the list so
                # model.register_hooks_for_layers can resolve it
                pass
            else:
                self.layer_names = ["embed"]

        if isinstance(sample_emb, list):
            first_shapes = [tuple(t.shape[1:]) for t in sample_emb]
        elif isinstance(sample_emb, dict):
            first_shapes = [
                tuple(v.shape[1:]) for k, v in sample_emb.items() if k in self.layer_names
            ]
        else:
            first_shapes = [tuple(sample_emb.shape[1:])]
        # Stash preliminary dims (may be refined later)
        self.embedding_dims = first_shapes

        num_samples = len(dataloader.dataset)
        use_streaming = (
            self._should_stream(first_shapes, num_samples)
            or self.config.use_streaming_embeddings is True
        )
        logger.info(f"Strategy decision: {'streaming' if use_streaming else 'in-memory'}")

        if use_streaming:
            dims = _extract_embeddings_streaming(
                base_model,
                dataloader,
                self.layer_names,
                device,
                save_path=save_path,
                chunk_size=self.config.chunk_size,
                compression=self.config.compression,
                compression_level=self.config.compression_level,
                aggregation=self.aggregation,
                auto_chunk_size=self.config.auto_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size,
                batch_chunk_size=self.config.batch_chunk_size,
                disable_tqdm=self.config.disable_tqdm,
                disable_layerdrop=self.config.disable_layerdrop,
            )
            # Persist final dims from streaming path
            try:
                self.embedding_dims = [tuple(d) for d in dims]
            except Exception:
                pass
            # Read num_labels from written HDF5
            try:
                with h5py.File(str(save_path), "r") as h5f:
                    nl = h5f.attrs.get("num_labels", None)
                    if nl is not None:
                        self.num_labels = int(nl)
            except Exception:
                pass
            # Hooks are internally managed by streaming util; ensure cleanup
            if hasattr(base_model, "deregister_all_hooks"):
                try:
                    base_model.deregister_all_hooks()
                except Exception:
                    pass
            # Respect requested layer_names; allow auto-detect when 'all'
            return HDF5EmbeddingDataset(
                save_path,
                layer_names=None if ("all" in self.layer_names) else self.layer_names,
                cache_in_memory=True,
                cache_size_limit_gb=max(1.0, float(self.config.cache_size_limit_gb)),
                allow_partial_cache=True,
            )

        # In-memory path
        embeds_dict, labels, dims_mem = _extract_embeddings_in_memory(
            base_model,
            dataloader,
            self.layer_names,
            device,
            aggregation=self.aggregation,
            disable_tqdm=self.config.disable_tqdm,
            disable_layerdrop=self.config.disable_layerdrop,
        )
        # Persist dims from in-memory path
        try:
            self.embedding_dims = [tuple(d) for d in dims_mem]
        except Exception:
            pass

        # Persist to disk for reuse (HDF5) when in-memory path is used
        try:
            # Compute num_labels from labels (handle one-hot)
            if isinstance(labels, torch.Tensor):
                if labels.dim() == 1:
                    num_labels = int(labels.unique().numel())
                else:
                    num_labels = int(torch.argmax(labels, dim=-1).unique().numel())
            else:
                num_labels = None  # Should not happen; keep guard
            save_embeddings_arrays(
                embeds_dict,
                labels,
                self.config.save_path,
                num_labels if num_labels is not None else 0,
                compression=self.config.compression,
                compression_level=self.config.compression_level,
            )
            logger.info(f"Saved embeddings to {self.config.save_path}")
            self.num_labels = num_labels
        except Exception as e:
            logger.warning(f"Failed to save embeddings to disk: {e}")
        if hasattr(base_model, "deregister_all_hooks"):
            try:
                base_model.deregister_all_hooks()
            except Exception:
                pass
        return EmbeddingDataset(embeds_dict, labels)
