"""Embedding extraction and management utilities.

This module provides utilities for extracting, storing, and loading embeddings
from representation learning models for offline analysis and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from esp_data.io import anypath
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


def _extract_embeddings_in_memory(
    model: ModelBase,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    aggregation: str = "mean",
    disable_tqdm: bool = False,
    disable_layerdrop: Optional[bool] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Original in-memory embedding extraction (kept for backward compatibility).

    Returns
    -------
    Tuple[Dict[str, torch.Tensor], torch.Tensor]
        (embeddings, labels) stacked on CPU. embeddings is a dictionary with layer names
        as keys and corresponding embedding tensors as values.

    Raises
    ------
    ValueError
        If no data is processed or dataloader is empty.
    """
    # Temporarily override model's disable_layerdrop if provided
    original_disable_layerdrop = None
    if disable_layerdrop is not None and hasattr(model, "disable_layerdrop"):
        original_disable_layerdrop = model.disable_layerdrop
        model.disable_layerdrop = disable_layerdrop

    # Dictionary to store embeddings for each layer
    layer_embeds: Dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []

    if not disable_tqdm:
        progress = tqdm(
            enumerate(dataloader),
            desc="Extracting embeddings (in-memory)",
            total=len(dataloader),
            unit="batch",
        )
    else:
        progress = enumerate(dataloader)

    logger.info(f"Extracting embeddings for {len(dataloader)} batches (in-memory mode)")

    try:
        with torch.no_grad():
            # Register hooks for the specified layers outside the loop
            layer_names = model.register_hooks_for_layers(layer_names)

            for _idx, batch in progress:
                wav = batch["raw_wav"].to(device)
                mask = batch.get("padding_mask")
                if mask is not None:
                    mask = mask.to(device)
                if mask is None:
                    emb = model.extract_embeddings(wav, aggregation=aggregation)
                else:
                    inp = {"raw_wav": wav, "padding_mask": mask}
                    emb = model.extract_embeddings(inp, aggregation=aggregation)

                # Log embedding shapes after extraction
                if isinstance(emb, list):
                    logger.debug(
                        f"Extracted embeddings (list of {len(emb)} tensors) with "
                        f"aggregation='{aggregation}':"
                    )
                    for i, layer_emb in enumerate(emb):
                        logger.debug(f"  Layer {i} shape: {layer_emb.shape}")
                elif isinstance(emb, dict):
                    logger.debug(
                        f"Extracted embeddings (dict with {len(emb)} layers) with "
                        f"aggregation='{aggregation}':"
                    )
                    for layer_name, layer_emb in emb.items():
                        logger.debug(f"  {layer_name} shape: {layer_emb.shape}")
                else:
                    logger.debug(
                        f"Extracted embeddings (single tensor) with "
                        f"aggregation='{aggregation}': {emb.shape}"
                    )

                # Handle single tensor, list of tensors, or dictionary from model
                if isinstance(emb, list):
                    # Multiple layers: emb is a list of tensors
                    for i, layer_emb in enumerate(emb):
                        layer_name = (
                            layer_names[i] if i < len(layer_names) else f"layer_{i}"
                        )
                        if layer_name not in layer_embeds:
                            layer_embeds[layer_name] = []
                        layer_embeds[layer_name].append(layer_emb.cpu())
                elif isinstance(emb, dict):
                    # Multiple layers: emb is a dictionary with layer names as keys
                    for layer_name, layer_emb in emb.items():
                        if layer_name not in layer_embeds:
                            layer_embeds[layer_name] = []
                        layer_embeds[layer_name].append(layer_emb.cpu())
                else:
                    # Single layer: emb is a single tensor
                    # Use the first layer name or a default name
                    layer_name = layer_names[0] if layer_names else "embeddings"
                    if layer_name not in layer_embeds:
                        layer_embeds[layer_name] = []
                    layer_embeds[layer_name].append(emb.cpu())

                labels.append(batch["label"].cpu())

        logger.debug(f"Extracted embeddings for {len(layer_embeds)} layers")

        # Check if we have any data
        if not labels:
            raise ValueError(
                "No data processed. Check if dataloader is empty or has "
                "invalid batches."
            )

        # Concatenate embeddings for each layer
        final_embeddings = {}
        embedding_dims = []
        for layer_name, layer_tensors in layer_embeds.items():
            final_embeddings[layer_name] = torch.cat(layer_tensors)
            embedding_dims.append(tuple(final_embeddings[layer_name].shape[1:]))

        return final_embeddings, torch.cat(labels), embedding_dims

    finally:
        # Restore original disable_layerdrop if it was overridden
        if original_disable_layerdrop is not None and hasattr(
            model, "disable_layerdrop"
        ):
            model.disable_layerdrop = original_disable_layerdrop
        # Always deregister hooks when done
        model.deregister_all_hooks()


def _extract_embeddings_streaming(
    model: ModelBase,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    save_path: Path,
    chunk_size: int,
    compression: str,
    compression_level: int,
    aggregation: str = "mean",
    auto_chunk_size: bool = True,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100,
    batch_chunk_size: int = 10,
    disable_tqdm: bool = False,
    disable_layerdrop: Optional[bool] = None,
) -> List[Tuple[int]]:
    """Memory-efficient streaming embedding extraction that saves to disk in chunks.

    This hybrid approach processes multiple batches in memory before writing to disk,
    significantly reducing I/O overhead while maintaining memory efficiency.

    Returns
    -------
    List[Tuple[int]]

    Raises
    ------
    ValueError
        If invalid parameters are provided.
    """
    # Temporarily override model's disable_layerdrop if provided
    original_disable_layerdrop = None
    if disable_layerdrop is not None and hasattr(model, "disable_layerdrop"):
        original_disable_layerdrop = model.disable_layerdrop
        model.disable_layerdrop = disable_layerdrop

    # Register hooks for the specified layers outside the loop
    layer_names = model.register_hooks_for_layers(layer_names)

    logger.info(
        f"Extracting embeddings using optimized streaming approach to {save_path}"
    )
    logger.info(f"Chunk size: {chunk_size}, layers: {len(layer_names)}")

    # Ensure directory exists
    save_path_obj = anypath(save_path)
    if not (hasattr(save_path_obj, "is_cloud") and save_path_obj.is_cloud):
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Get first batch to determine embedding dimensions
    first_batch = next(iter(dataloader))
    wav = first_batch["raw_wav"].to(device)
    mask = first_batch.get("padding_mask")
    if mask is not None:
        mask = mask.to(device)

    if mask is None:
        sample_emb = model.extract_embeddings(wav, aggregation=aggregation)
    else:
        inp = {"raw_wav": wav, "padding_mask": mask}
        sample_emb = model.extract_embeddings(inp, aggregation=aggregation)

    # Handle single tensor, list of tensors, or dictionary
    if isinstance(sample_emb, list):
        # For multi-layer, collect dimensions for each layer
        # Exclude batch dimension (0)
        embedding_dims = [emb.shape[1:] for emb in sample_emb]
        logger.info(f"Multi-layer embedding dimensions: {embedding_dims}")
    elif isinstance(sample_emb, dict):
        # For multi-layer dictionary, collect dimensions for each layer
        # Exclude batch dimension (0)
        embedding_dims = [emb.shape[1:] for emb in sample_emb.values()]
        logger.info(f"Multi-layer embedding dimensions (dict): {embedding_dims}")
    else:
        # Single tensor case - still store as list for consistency
        embedding_dims = [sample_emb.shape[1:]]  # Exclude batch dimension
        logger.info(f"Single-layer embedding dimensions: {embedding_dims}")

    total_samples = len(dataloader.dataset)
    logger.info(f"Total samples: {total_samples}")

    # Calculate total embedding size per sample (for HDF5 and auto-chunking
    # calculations)
    total_embedding_size = int(sum(np.prod(dim) for dim in embedding_dims))

    # Calculate optimal chunk size if auto-calculation is enabled
    if auto_chunk_size and torch.cuda.is_available():
        try:
            available_memory = torch.cuda.get_device_properties(0).total_memory
            # Calculate optimal chunk size based on available memory
            # Estimate memory per sample: sum of all embedding dimensions * 4 bytes
            # (float32) + overhead
            # Add 1KB overhead per sample
            memory_per_sample = total_embedding_size * 4 + 1024
            optimal_chunk_size = int(
                available_memory * 0.95 / memory_per_sample
            )  # Use 95% of GPU memory

            # Apply min/max constraints
            chunk_size = max(min_chunk_size, min(max_chunk_size, optimal_chunk_size))
            logger.info(
                f"Auto-calculated chunk size: {chunk_size} (from {optimal_chunk_size})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to auto-calculate chunk size: {e}, "
                f"using provided value: {chunk_size}"
            )

    # Use provided batch chunk size or calculate optimal one
    if batch_chunk_size <= 0:
        # Calculate batch chunk size - process multiple batches before writing to disk
        # This reduces I/O overhead while keeping memory usage reasonable
        avg_batch_size = (
            total_samples // len(dataloader) if len(dataloader) > 0 else chunk_size
        )
        batch_chunk_size = max(5, min(20, chunk_size // max(1, avg_batch_size // 10)))
        logger.info(
            f"Auto-calculated batch chunk size: {batch_chunk_size} "
            f"(avg batch size: {avg_batch_size})"
        )
    else:
        logger.info(f"Using provided batch chunk size: {batch_chunk_size}")

    # Validate chunk size against HDF5 limits
    # HDF5 has a 4GB chunk size limit: chunk_size * total_embedding_size * 4 bytes < 4GB
    max_chunk_size_for_hdf5 = int(
        4 * 1024 * 1024 * 1024 / max(1, (total_embedding_size * 4))
    )  # 4GB limit (guard against zero)
    if chunk_size > max_chunk_size_for_hdf5:
        logger.warning(
            f"Chunk size {chunk_size} exceeds HDF5 4GB limit for "
            f"total embedding size {total_embedding_size}. "
            f"Adjusting to {max_chunk_size_for_hdf5}."
        )
        chunk_size = max_chunk_size_for_hdf5

    # Validate chunk size
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk size: {chunk_size}. Must be positive.")
    if chunk_size > total_samples:
        logger.warning(
            f"Chunk size {chunk_size} is larger than total samples "
            f"{total_samples}. Adjusting to {total_samples}."
        )
        chunk_size = total_samples

    try:
        # Create HDF5 file with streaming approach
        if hasattr(save_path_obj, "is_cloud") and save_path_obj.is_cloud:
            with save_path_obj.open("wb") as fh, h5py.File(fh, "w") as h5f:
                _create_and_fill_h5_datasets_hybrid(
                    h5f,
                    total_samples,
                    embedding_dims,
                    chunk_size,
                    compression,
                    compression_level,
                    model,
                    dataloader,
                    layer_names,
                    device,
                    first_batch,
                    batch_chunk_size,
                    aggregation,
                    disable_tqdm,
                )
        else:
            with h5py.File(str(save_path_obj), "w") as h5f:
                _create_and_fill_h5_datasets_hybrid(
                    h5f,
                    total_samples,
                    embedding_dims,
                    chunk_size,
                    compression,
                    compression_level,
                    model,
                    dataloader,
                    layer_names,
                    device,
                    first_batch,
                    batch_chunk_size,
                    aggregation,
                    disable_tqdm,
                )

        return [tuple(emb) for emb in embedding_dims]

    finally:
        # Restore original disable_layerdrop if it was overridden
        if original_disable_layerdrop is not None and hasattr(
            model, "disable_layerdrop"
        ):
            model.disable_layerdrop = original_disable_layerdrop
        # Always deregister hooks when done
        model.deregister_all_hooks()


def _create_and_fill_h5_datasets_hybrid(
    h5f: h5py.File,
    total_samples: int,
    embedding_dims: List[tuple],
    chunk_size: int,
    compression: str,
    compression_level: int,
    model: ModelBase,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    first_batch: dict,
    batch_chunk_size: int,
    aggregation: str = "mean",
    disable_tqdm: bool = False,
) -> None:
    """Helper function to create and fill HDF5 datasets with optimized hybrid approach.

    All embeddings are saved as numpy.float32 for maximum precision and HDF5
    compatibility.
    All labels are saved as numpy.int64 for standard integer labels.

    Raises
    ------
    ValueError
        If invalid parameters or data structures are encountered.
    """
    # Get label shape and type from the first batch
    sample_labels = first_batch["label"]

    # Validate that labels exist and have the expected structure
    if sample_labels is None:
        raise ValueError("Labels not found in batch")

    label_shape = sample_labels.shape[1:] if sample_labels.dim() > 1 else ()
    label_dtype = sample_labels.dtype

    logger.info(
        f"Sample labels info: shape={sample_labels.shape}, "
        f"dims={sample_labels.dim()}, dtype={label_dtype}"
    )

    # Convert PyTorch dtype to numpy dtype for HDF5
    if hasattr(label_dtype, "numpy_dtype"):
        numpy_label_dtype = label_dtype.numpy_dtype
    else:
        # Handle common PyTorch dtypes
        dtype_mapping = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        numpy_label_dtype = dtype_mapping.get(label_dtype)
        if numpy_label_dtype is None:
            # If we can't map the dtype, log a warning and use a safe default
            logger.warning(
                f"Unknown PyTorch dtype: {label_dtype}. Using np.float32 as fallback."
            )
            numpy_label_dtype = np.float32

    # Validate that we got a valid numpy dtype
    if numpy_label_dtype is None:
        raise ValueError(
            f"Could not convert PyTorch dtype {label_dtype} to numpy dtype"
        )

    logger.info(
        f"Label shape: {sample_labels.shape}, dtype: {label_dtype} -> "
        f"numpy: {numpy_label_dtype}"
    )

    # Create separate datasets for each layer to handle different dimensions
    embeddings_datasets = {}
    if aggregation == "mean" and len(embedding_dims) == 1:
        # For mean aggregation with single concatenated tensor, create one dataset
        layer_name = layer_names[0] if layer_names else "embeddings"
        embedding_dim = embedding_dims[0]
        h5_shape = (total_samples,) + embedding_dim
        h5_chunks = (chunk_size,) + embedding_dim

        embeddings_datasets[layer_name] = h5f.create_dataset(
            f"embeddings_{layer_name}",
            shape=h5_shape,
            maxshape=(None,) + embedding_dim,
            dtype=np.float32,
            chunks=h5_chunks,
            compression=compression,
            compression_opts=compression_level,
        )
    else:
        # For multi-layer or none aggregation, create datasets for each layer
        for layer_name, embedding_dim in zip(layer_names, embedding_dims, strict=False):
            # Create HDF5 dataset with original multi-dimensional shape
            h5_shape = (total_samples,) + embedding_dim
            h5_chunks = (chunk_size,) + embedding_dim

            embeddings_datasets[layer_name] = h5f.create_dataset(
                f"embeddings_{layer_name}",
                shape=h5_shape,
                maxshape=(None,) + embedding_dim,
                dtype=np.float32,
                chunks=h5_chunks,
                compression=compression,
                compression_opts=compression_level,
            )

    # Create labels dataset with correct shape
    if label_shape:
        # Multi-dimensional labels (e.g., one-hot encoded)
        labels_dset = h5f.create_dataset(
            "labels",
            shape=(total_samples,) + label_shape,
            maxshape=(None,) + label_shape,
            dtype=numpy_label_dtype,
            chunks=(chunk_size,) + label_shape,
            compression=compression,
            compression_opts=compression_level,
        )
    else:
        # Single-dimensional labels
        labels_dset = h5f.create_dataset(
            "labels",
            shape=(total_samples,),
            maxshape=(None,),
            dtype=numpy_label_dtype,
            chunks=(chunk_size,),
            compression=compression,
            compression_opts=compression_level,
        )

    # Log dataset shapes
    embedding_shapes = {name: dset.shape for name, dset in embeddings_datasets.items()}
    logger.info(
        f"Dataset shapes - embeddings: {embedding_shapes}, labels: {labels_dset.shape}"
    )
    logger.info(f"Starting hybrid extraction with {total_samples} total samples")
    logger.info(
        f"Chunk size: {chunk_size}, batch chunk size: {batch_chunk_size}, "
        f"compression: {compression} (level {compression_level})"
    )

    # Process in chunks to save memory
    start_idx = 0

    # Process the first batch separately since we already have it
    with torch.no_grad():
        # Process first batch
        wav = first_batch["raw_wav"].to(device)
        mask = first_batch.get("padding_mask")
        if mask is not None:
            mask = mask.to(device)

        if mask is None:
            embeddings = model.extract_embeddings(wav, aggregation=aggregation)
        else:
            inp = {"raw_wav": wav, "padding_mask": mask}
            embeddings = model.extract_embeddings(inp, aggregation=aggregation)

        # Determine batch size from tensor shapes, not number of layers
        if isinstance(embeddings, list):
            if len(embeddings) == 0:
                raise ValueError("Empty embeddings list returned for first batch")
            batch_size = int(embeddings[0].shape[0])
        else:
            batch_size = int(embeddings.shape[0])

        # Validate first batch label shape
        first_batch_labels = first_batch["label"]
        if first_batch_labels.shape[1:] != label_shape:
            raise ValueError(
                f"First batch label shape {first_batch_labels.shape} doesn't match "
                f"expected shape {label_shape}"
            )

        # Validate batch size consistency
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size: {batch_size}. Must be positive.")

        # Write first batch to HDF5
        try:
            # Handle single tensor, list of tensors, or dictionary
            if isinstance(embeddings, list):
                # Multi-layer case: write each layer to its own dataset
                for _i, (layer_name, layer_emb) in enumerate(
                    zip(layer_names, embeddings, strict=False)
                ):
                    layer_emb_np = layer_emb.cpu().numpy().astype(np.float32)
                    embeddings_datasets[layer_name][
                        start_idx : start_idx + batch_size
                    ] = layer_emb_np
            elif isinstance(embeddings, dict):
                # Multi-layer dictionary case: write each layer to its own dataset
                for layer_name, layer_emb in embeddings.items():
                    layer_emb_np = layer_emb.cpu().numpy().astype(np.float32)
                    embeddings_datasets[layer_name][
                        start_idx : start_idx + batch_size
                    ] = layer_emb_np
            else:
                # Single tensor case: write to first available dataset
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                first_layer_name = list(embeddings_datasets.keys())[0]
                embeddings_datasets[first_layer_name][
                    start_idx : start_idx + batch_size
                ] = embeddings_np

            labels_np = first_batch_labels.numpy().astype(np.int64)
            labels_dset[start_idx : start_idx + batch_size] = labels_np
        except Exception as e:
            logger.error(f"Error writing first batch to HDF5: {e}")
            raise

        start_idx += batch_size
        logger.info(
            f"Processed first batch: {batch_size} samples, "
            f"total processed: {start_idx}/{total_samples}"
        )

        # Clear memory
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Process remaining batches using hybrid approach
        # IMPORTANT: Skip the first batch, which we already processed above
        data_iter = iter(dataloader)
        try:
            _ = next(data_iter)
        except StopIteration:
            data_iter = iter(())
        if not disable_tqdm:
            progress = tqdm(
                enumerate(data_iter),
                desc="Extracting embeddings (hybrid streaming)",
                total=max(0, len(dataloader) - 1),
                unit="batch",
            )
        else:
            progress = enumerate(data_iter)

        # Buffers for collecting multiple batches before writing
        batch_embeddings_buffer = []
        batch_labels_buffer = []
        buffer_sample_count = 0

        for batch_idx, batch in progress:
            logger.debug(f"Processing batch {batch_idx}")
            wav = batch["raw_wav"].to(device)
            mask = batch.get("padding_mask")
            if mask is not None:
                mask = mask.to(device)

            if mask is None:
                embeddings = model.extract_embeddings(wav, aggregation=aggregation)
            else:
                inp = {"raw_wav": wav, "padding_mask": mask}
                embeddings = model.extract_embeddings(inp, aggregation=aggregation)

            shapes_description = (
                [e.shape for e in embeddings]
                if isinstance(embeddings, list)
                else embeddings.shape
            )
            logger.debug(
                (
                    f"Batch {batch_idx} embeddings type: {type(embeddings)}, "
                    f"shape: {shapes_description}"
                )
            )

            # Determine batch size from tensor shapes, not number of layers
            if isinstance(embeddings, list):
                if len(embeddings) == 0:
                    raise ValueError(
                        f"Empty embeddings list returned in batch {batch_idx}"
                    )
                batch_size = int(embeddings[0].shape[0])
            elif isinstance(embeddings, dict):
                if len(embeddings) == 0:
                    raise ValueError(
                        f"Empty embeddings dictionary returned in batch {batch_idx}"
                    )
                batch_size = int(next(iter(embeddings.values())).shape[0])
            else:
                batch_size = int(embeddings.shape[0])

            # Validate batch size consistency
            if batch_size <= 0:
                raise ValueError(
                    f"Invalid batch size in batch {batch_idx}: {batch_size}. "
                    f"Must be positive."
                )

            # Validate batch label shape
            batch_labels = batch["label"]
            if batch_labels.shape[1:] != label_shape:
                raise ValueError(
                    f"Batch {batch_idx} label shape {batch_labels.shape} doesn't match "
                    f"expected shape {label_shape}"
                )

            # Add to buffer - handle single tensor, list of tensors, or dictionary
            if isinstance(embeddings, list):
                # For aggregation="none", embeddings is a list of tensors
                batch_embeddings_buffer.append([emb.cpu() for emb in embeddings])
            elif isinstance(embeddings, dict):
                # For multi-layer dictionary, convert to list of tensors
                batch_embeddings_buffer.append(
                    [emb.cpu() for emb in embeddings.values()]
                )
            else:
                # For other aggregation methods, embeddings is a single tensor
                batch_embeddings_buffer.append(embeddings.cpu())
            batch_labels_buffer.append(batch_labels)
            buffer_sample_count += batch_size

            # Write to disk when buffer is full or at the end
            should_write = (
                buffer_sample_count >= chunk_size
                or batch_idx == len(dataloader) - 1
                or len(batch_embeddings_buffer) >= batch_chunk_size
            )

            if should_write and batch_embeddings_buffer:
                # Concatenate all buffered batches
                try:
                    # Concatenate labels
                    if len(batch_labels_buffer) == 1:
                        concatenated_labels = batch_labels_buffer[0]
                    else:
                        concatenated_labels = torch.cat(batch_labels_buffer, dim=0)

                    # Convert labels to numpy
                    labels_np = concatenated_labels.numpy().astype(np.int64)

                    # Validate we won't exceed dataset bounds
                    remaining_samples = 0  # Initialize to 0
                    if start_idx + len(labels_np) > total_samples:
                        logger.warning(
                            "Buffer would exceed dataset bounds. Adjusting buffer size."
                        )
                        remaining_samples = total_samples - start_idx
                        if remaining_samples > 0:
                            labels_np = labels_np[:remaining_samples]
                        else:
                            break

                    # Handle embeddings - check if first item is a list (multi-layer)
                    if isinstance(batch_embeddings_buffer[0], list):
                        # Multi-layer case: concatenate each layer separately across
                        # batches
                        num_layers = len(batch_embeddings_buffer[0])
                        logger.debug(
                            (
                                f"Writing {num_layers} layers from buffer, "
                                f"layer_names: {layer_names}"
                            )
                        )
                        for layer_idx in range(num_layers):
                            layer_embeddings = [
                                batch_emb[layer_idx]
                                for batch_emb in batch_embeddings_buffer
                            ]
                            if len(layer_embeddings) == 1:
                                concatenated_layer = layer_embeddings[0]
                            else:
                                concatenated_layer = torch.cat(layer_embeddings, dim=0)

                            layer_emb_np = concatenated_layer.numpy().astype(np.float32)

                            # Adjust size if needed
                            if remaining_samples > 0:
                                layer_emb_np = layer_emb_np[:remaining_samples]

                            layer_name = layer_names[layer_idx]
                            logger.debug(
                                (
                                    f"Writing layer {layer_idx} ({layer_name}) with "
                                    f"shape {layer_emb_np.shape} to dataset "
                                    f"{embeddings_datasets[layer_name].shape}"
                                )
                            )
                            embeddings_datasets[layer_name][
                                start_idx : start_idx + len(layer_emb_np)
                            ] = layer_emb_np
                    else:
                        # Single tensor case: concatenate batches and write to first
                        # available dataset
                        if len(batch_embeddings_buffer) == 1:
                            concatenated_embeddings = batch_embeddings_buffer[0]
                        else:
                            concatenated_embeddings = torch.cat(
                                batch_embeddings_buffer, dim=0
                            )

                        embeddings_np = concatenated_embeddings.numpy().astype(
                            np.float32
                        )

                        # Adjust size if needed
                        if remaining_samples > 0:
                            embeddings_np = embeddings_np[:remaining_samples]

                        first_layer_name = list(embeddings_datasets.keys())[0]
                        embeddings_datasets[first_layer_name][
                            start_idx : start_idx + len(embeddings_np)
                        ] = embeddings_np

                    # Write labels
                    labels_dset[start_idx : start_idx + len(labels_np)] = labels_np

                    start_idx += len(labels_np)

                    logger.debug(
                        f"Wrote buffer to disk - samples: {len(labels_np)}, "
                        f"total processed: {start_idx}/{total_samples}"
                    )

                except Exception as e:
                    logger.error(f"Error writing buffer to HDF5: {e}")
                    raise

                # Clear buffers
                batch_embeddings_buffer.clear()
                batch_labels_buffer.clear()
                buffer_sample_count = 0

                # Clear memory (guard variables depending on path)
                try:
                    del concatenated_embeddings
                except Exception:
                    pass
                try:
                    del concatenated_labels
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force flush to disk periodically (less frequent now)
                if batch_idx % 100 == 0:
                    h5f.flush()

        # Clear any remaining items in buffers
        del batch_embeddings_buffer, batch_labels_buffer

    # Add metadata - handle multi-dimensional labels properly
    if label_shape:
        # For multi-dimensional labels, we need to determine num_labels differently
        if len(label_shape) == 1 and label_shape[0] > 1:
            # Likely one-hot encoded, use the dimension as num_labels
            h5f.attrs["num_labels"] = label_shape[0]
        else:
            # For other multi-dimensional cases, store the shape info
            h5f.attrs["num_labels"] = -1  # Indicate multi-dimensional
            h5f.attrs["label_shape"] = label_shape
    else:
        # Single-dimensional labels - use numpy operations to avoid PyTorch
        # conversion issues
        unique_labels = np.unique(labels_dset[:])
        h5f.attrs["num_labels"] = len(unique_labels)

    # Add multi-layer metadata. We always save embeddings in per-layer datasets
    # (embeddings_{layer}) even if there is only one layer, so mark as multi-layer.
    h5f.attrs["multi_layer"] = True
    # Set layer_names to only the layers that were actually created
    actual_layer_names = list(embeddings_datasets.keys())
    h5f.attrs["layer_names"] = actual_layer_names
    # Convert embedding_dims to a format HDF5 can store
    h5f.attrs["embedding_dims"] = [str(tuple(dim)) for dim in embedding_dims]

    # Final validation
    if start_idx != total_samples:
        logger.warning(
            f"Expected to process {total_samples} samples, but processed {start_idx}"
        )
        if start_idx < total_samples:
            logger.warning(
                f"Missing {total_samples - start_idx} samples. "
                f"This might indicate an issue."
            )

    logger.info(f"Completed hybrid extraction. Total samples processed: {start_idx}")
    # Store metadata as HDF5 attributes (reinforce values, do not change multi_layer)
    # Don't overwrite layer_names - it was already set correctly above

    final_embedding_shapes = {
        name: dset.shape for name, dset in embeddings_datasets.items()
    }
    logger.info(
        f"Final dataset shapes - embeddings: {final_embedding_shapes}, "
        f"labels: {labels_dset.shape}"
    )


class EmbeddingDataset(torch.utils.data.Dataset):
    """Simple dataset that serves pre-computed embeddings and labels."""

    def __init__(
        self,
        embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels

        # Compute embedding dimensions for consistency with HDF5EmbeddingDataset
        if isinstance(embeddings, dict):
            self.embedding_dims = [tuple(emb.shape[1:]) for emb in embeddings.values()]
        else:
            self.embedding_dims = [tuple(embeddings.shape[1:])]

    def __len__(self) -> int:  # noqa: D401
        if isinstance(self.embeddings, dict):
            # Get length from first layer (all should have same length)
            first_layer = next(iter(self.embeddings.values()))
            return first_layer.size(0)
        else:
            return self.embeddings.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: D401
        if isinstance(self.embeddings, dict):
            # Multi-layer embeddings: return dict with layer names as keys
            return {
                **{
                    layer_name: layer_emb[idx]
                    for layer_name, layer_emb in self.embeddings.items()
                },
                "label": self.labels[idx],
            }
        else:
            # Single tensor: return as before for backward compatibility
            return {"embed": self.embeddings[idx], "label": self.labels[idx]}


class HDF5EmbeddingDataset(torch.utils.data.Dataset):
    """Optimized HDF5 dataset with proper multi-worker support.

    Key improvements:
    - Per-worker file handles (fixes multi-worker bottleneck)
    - Lazy file opening (only opens when needed)
    - Proper cleanup and file handle management
    - Efficient indexing with minimal overhead
    - Optional in-memory caching for small datasets
    """

    def __init__(
        self,
        hdf5_path: Union[str, Path],
        layer_names: Optional[List[str]] = None,
        cache_in_memory: bool = True,
        cache_size_limit_gb: float = 8.0,
        allow_partial_cache: bool = True,
    ) -> None:
        """Initialize the optimized HDF5 embedding dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing embeddings and labels
            layer_names: Optional list of layer names to use. If None, will be
                auto-detected from the HDF5 file
            cache_in_memory: If True and dataset is small enough, cache everything
                in RAM
            cache_size_limit_gb: Maximum dataset size (GB) to allow in-memory caching
        """
        self.hdf5_path = Path(hdf5_path)
        self.cache_in_memory = cache_in_memory
        self.allow_partial_cache = allow_partial_cache

        # File handle will be opened per-worker
        self._h5file = None
        self._datasets = None
        self._worker_id = None

        # Load metadata from a temporary file handle
        self._load_metadata()

        # Handle layer names
        if layer_names is None:
            self.layer_names = self._detect_layer_names()
        else:
            # Check if 'last_layer' needs to be resolved
            if "last_layer" in layer_names:
                detected_layers = self._detect_layer_names()
                if detected_layers:
                    # Replace 'last_layer' with the actual last layer name
                    self.layer_names = [
                        name if name != "last_layer" else detected_layers[-1]
                        for name in layer_names
                    ]
                    logger.info(f"Resolved 'last_layer' to '{detected_layers[-1]}'")
                else:
                    self.layer_names = layer_names
                    logger.warning("No layer names found to resolve 'last_layer'")
            else:
                self.layer_names = layer_names

        # Compute embedding dimensions
        self.embedding_dims = self._compute_embedding_dims()

        # Compute num_labels
        self.num_labels = self._compute_num_labels()

        # In-memory cache
        self._memory_cache: Optional[Dict[str, torch.Tensor]] = None
        if cache_in_memory:
            self._try_cache_in_memory(cache_size_limit_gb)

        # Sliding window cache (sample-wise) if full/partial layer cache not available
        # Always available for fallback when embeddings not fully cached
        self._window_cache: Optional[Dict[str, torch.Tensor]] = None
        self._window_start: int = 0
        self._window_end: int = 0
        self._window_capacity_samples: int = 0
        self._cache_size_limit_gb: float = float(cache_size_limit_gb)

        # Initialize sliding window capacity if embeddings not fully cached
        if (
            self._memory_cache is None
            or (
                self.multi_layer
                and not all(ln in self._memory_cache for ln in self.layer_names)
            )
            or (not self.multi_layer and "embed" not in (self._memory_cache or {}))
        ):
            self._initialize_window_cache()

        logger.info(
            f"Initialized HDF5EmbeddingDataset: {len(self)} samples, "
            f"layers: {self.layer_names}"
        )
        logger.info(f"Embedding dimensions: {self.embedding_dims}")
        logger.info(f"Number of labels: {self.num_labels}")
        logger.info(f"In-memory cache: {self._memory_cache is not None}")

    def _bytes_per_sample_embeddings(self) -> int:
        """Compute bytes required to store embeddings for a single sample.

        Returns:
            int: Number of bytes required for a single sample.
        """
        if self.multi_layer:
            total = 0
            for dims in self.embedding_dims:
                # dims per layer
                if isinstance(dims, tuple) and len(dims) > 0:
                    numel = 1
                    for d in dims:
                        numel *= int(d)
                    total += numel * 4  # float32
            return int(total)
        else:
            dims = self.embedding_dims[0] if self.embedding_dims else ()
            if isinstance(dims, tuple) and len(dims) > 0:
                numel = 1
                for d in dims:
                    numel *= int(d)
                return int(numel * 4)
            return 0

    def _initialize_window_cache(self) -> None:
        """Initialize sliding window cache capacity and prefill the first window."""
        try:
            bytes_per_sample = self._bytes_per_sample_embeddings()
            if bytes_per_sample <= 0:
                self._window_capacity_samples = 0
                return
            budget_bytes = int(self._cache_size_limit_gb * (1024**3))
            # Reserve a small margin for overhead
            budget_bytes = max(0, budget_bytes - 8 * 1024 * 1024)
            capacity = max(1, budget_bytes // bytes_per_sample)
            self._window_capacity_samples = int(min(capacity, self.total_samples))
            # Prefill initial window from start
            if self._window_capacity_samples > 0:
                self._load_window(0)
        except Exception as e:
            logger.warning(f"Failed to initialize window cache: {e}")
            self._window_capacity_samples = 0

    def _load_window(self, start_idx: int) -> None:
        """Load a new cache window [start_idx, start_idx + capacity)."""
        if self._window_capacity_samples <= 0:
            self._window_cache = None
            self._window_start = 0
            self._window_end = 0
            return
        # Ensure file handle available
        datasets = self._datasets
        if datasets is None:
            self._get_file_handle()
            datasets = self._datasets
        if datasets is None:
            return
        start = max(0, int(start_idx))
        end = min(self.total_samples, start + self._window_capacity_samples)
        if end <= start:
            return
        window: Dict[str, torch.Tensor] = {}
        # Load embeddings per layer/dataset
        if self.multi_layer:
            for layer_name in self.layer_names:
                if self._memory_cache is not None and layer_name in self._memory_cache:
                    # This layer already fully cached; no need to duplicate
                    continue
                if layer_name in datasets:
                    slice_np = datasets[layer_name][start:end]
                    window[layer_name] = torch.from_numpy(
                        np.asarray(slice_np, dtype=np.float32)
                    )
        else:
            if self._memory_cache is None or "embed" not in self._memory_cache:
                if "embed" in datasets:
                    slice_np = datasets["embed"][start:end]
                    window["embed"] = torch.from_numpy(
                        np.asarray(slice_np, dtype=np.float32)
                    )
        # Assign window
        self._window_cache = window if window else None
        self._window_start = start
        self._window_end = end

    def _get_file_handle(self) -> h5py.File:
        """Get or create file handle for current worker.

        This is crucial for multi-worker DataLoader support. Each worker
        needs its own file handle to avoid thread-safety issues.

        Returns:
            h5py.File: The file handle for the current worker.
        """
        import torch.utils.data

        # Get current worker info
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else None

        # If worker changed or file not open, (re)open file
        if self._h5file is None or self._worker_id != worker_id:
            # Close old file if exists
            if self._h5file is not None:
                try:
                    self._h5file.close()
                except Exception:
                    pass

            # Open new file handle
            self._h5file = h5py.File(self.hdf5_path, "r", swmr=True)
            self._worker_id = worker_id

            # Refresh dataset references
            self._datasets = self._get_dataset_references()

            if worker_id is not None:
                logger.debug(f"Opened HDF5 file in worker {worker_id}")

        return self._h5file

    def _load_metadata(self) -> None:
        """Load essential metadata from HDF5 file."""
        with h5py.File(self.hdf5_path, "r") as f:
            # Get total samples from labels
            self.total_samples = f["labels"].shape[0]

            # Check for multi-layer structure
            self.multi_layer = any(key.startswith("embeddings_") for key in f.keys())

            # Store all dataset keys for later use
            self._dataset_keys = list(f.keys())

            # Store HDF5 attributes
            self._h5_attrs = dict(f.attrs)

    def _detect_layer_names(self) -> List[str]:
        """Auto-detect layer names from dataset structure.

        Returns:
            List[str]: List of detected layer names.
        """
        if self.multi_layer:
            layer_names = []
            for key in self._dataset_keys:
                if key.startswith("embeddings_"):
                    layer_name = key.replace("embeddings_", "")
                    layer_names.append(layer_name)
            return sorted(layer_names)
        else:
            return ["embed"]

    def _get_dataset_references(self) -> Dict[str, h5py.Dataset]:
        """Get direct references to all embedding datasets.

        Returns:
            Dict[str, h5py.Dataset]: Dictionary mapping layer names to datasets.

        Raises:
            ValueError: If no datasets are found.
        """
        if self._h5file is None:
            self._get_file_handle()

        datasets = {}

        if self.multi_layer:
            for layer_name in self.layer_names:
                dataset_name = f"embeddings_{layer_name}"
                if dataset_name in self._h5file:
                    datasets[layer_name] = self._h5file[dataset_name]
                else:
                    logger.warning(f"Dataset {dataset_name} not found")
        else:
            if "embeddings" in self._h5file:
                datasets["embed"] = self._h5file["embeddings"]
            else:
                for key in self._h5file.keys():
                    if key.startswith("embeddings_"):
                        datasets["embed"] = self._h5file[key]
                        break
                else:
                    raise ValueError("No embedding datasets found")

        datasets["labels"] = self._h5file["labels"]
        return datasets

    def _compute_embedding_dims(self) -> List[Tuple[int, ...]]:
        """Compute embedding dimensions for each layer.

        Returns:
            List[Tuple[int, ...]]: List of embedding dimensions for each layer.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            embedding_dims = []

            if self.multi_layer:
                for layer_name in self.layer_names:
                    dataset_name = f"embeddings_{layer_name}"
                    if dataset_name in f:
                        shape = f[dataset_name].shape[1:]
                        embedding_dims.append(shape)
                    else:
                        embedding_dims.append(())
            else:
                if "embeddings" in f:
                    shape = f["embeddings"].shape[1:]
                else:
                    for key in f.keys():
                        if key.startswith("embeddings_"):
                            shape = f[key].shape[1:]
                            break
                    else:
                        shape = ()
                embedding_dims.append(shape)

            return embedding_dims

    def _compute_num_labels(self) -> Optional[int]:
        """Compute number of unique labels.

        Returns:
            Optional[int]: Number of unique labels, or None if not available.
        """
        try:
            if "num_labels" in self._h5_attrs:
                return int(self._h5_attrs["num_labels"])

            # Fallback: compute from a sample
            with h5py.File(self.hdf5_path, "r") as f:
                # Sample labels to avoid loading entire array
                sample_size = min(10000, f["labels"].shape[0])
                labels = f["labels"][:sample_size]

                if labels.ndim > 1 and labels.shape[-1] > 1:
                    labels = labels.argmax(axis=-1)

                return int(np.unique(labels).size)
        except Exception as e:
            logger.warning(f"Could not compute num_labels: {e}")
            return None

    def _try_cache_in_memory(self, size_limit_gb: float) -> None:
        """Try to cache as much of the dataset in memory as possible.

        Strategy
        - If the full dataset (all layers + labels) fits within ``size_limit_gb``,
          cache everything.
        - Otherwise, cache labels and as many embedding layers (entire tensors)
          as possible until the budget is exhausted (smallest layers first).
        """
        try:
            # Estimate dataset size
            with h5py.File(self.hdf5_path, "r") as f:
                total_bytes = 0

                # Calculate embedding sizes
                if self.multi_layer:
                    layer_sizes: Dict[str, int] = {}
                    for layer_name in self.layer_names:
                        dataset_name = f"embeddings_{layer_name}"
                        if dataset_name in f:
                            dset = f[dataset_name]
                            layer_sizes[layer_name] = int(
                                np.prod(dset.shape) * dset.dtype.itemsize
                            )
                            total_bytes += layer_sizes[layer_name]
                else:
                    if "embeddings" in f:
                        dset = f["embeddings"]
                    else:
                        dset = next(
                            f[k] for k in f.keys() if k.startswith("embeddings_")
                        )
                    total_bytes += np.prod(dset.shape) * dset.dtype.itemsize

                # Add labels size
                labels_bytes = int(
                    np.prod(f["labels"].shape) * f["labels"].dtype.itemsize
                )
                total_bytes += labels_bytes

                size_gb = total_bytes / (1024**3)
                logger.info(f"Dataset size: {size_gb:.2f} GB")

                # Initialize cache
                self._memory_cache = {}

                # Always cache labels first
                self._memory_cache["labels"] = torch.from_numpy(
                    f["labels"][:].astype(np.int64)
                )

                remaining_bytes = int(size_limit_gb * (1024**3)) - labels_bytes

                if remaining_bytes <= 0:
                    logger.info(
                        "Memory budget exhausted after caching labels; embeddings "
                        "will be streamed."
                    )
                    # Keep cache limited to labels
                    if not self.allow_partial_cache:
                        # If partial caching not allowed and total too big, drop cache
                        self._memory_cache = None
                    return

                if self.multi_layer:
                    # Sort layers by size (smallest first) and cache while budget allows
                    for layer_name in sorted(layer_sizes, key=lambda k: layer_sizes[k]):
                        if (
                            layer_sizes[layer_name] <= remaining_bytes
                            and f.get(f"embeddings_{layer_name}") is not None
                        ):
                            data_np = f[f"embeddings_{layer_name}"][:].astype(
                                np.float32
                            )
                            self._memory_cache[layer_name] = torch.from_numpy(data_np)
                            remaining_bytes -= layer_sizes[layer_name]
                            logger.info(
                                f"Cached layer '{layer_name}' in memory "
                                f"(size={layer_sizes[layer_name] / (1024**2):.2f} MB)"
                            )
                        else:
                            logger.info(
                                f"Skipping cache for layer '{layer_name}' due to "
                                f"memory budget"
                            )
                else:
                    # Single dataset - try to cache if within budget
                    if "embeddings" in f:
                        dset = f["embeddings"]
                    else:
                        key = next(k for k in f.keys() if k.startswith("embeddings_"))
                        dset = f[key]
                    bytes_needed = int(np.prod(dset.shape) * dset.dtype.itemsize)
                    if bytes_needed <= remaining_bytes:
                        data = dset[:].astype(np.float32)
                        self._memory_cache["embed"] = torch.from_numpy(data)
                        logger.info(
                            f"Cached single embedding dataset in memory "
                            f"(size={bytes_needed / (1024**2):.2f} MB)"
                        )
                    else:
                        logger.info(
                            "Skipping cache for embeddings due to memory budget; "
                            "will stream from HDF5"
                        )

        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")
            self._memory_cache = None

    def __len__(self) -> int:
        """Return the total number of samples.

        Returns:
            int: Total number of samples in the dataset.
        """
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with optimized data loading.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing embeddings and labels.
        """
        # Use memory cache if available
        if self._memory_cache is not None:
            if self.multi_layer:
                result = {"label": self._memory_cache["labels"][idx]}
                for layer_name in self.layer_names:
                    if layer_name in self._memory_cache:
                        result[layer_name] = self._memory_cache[layer_name][idx]
                    else:
                        # Try window cache first, otherwise fallback to on-demand read
                        if not (self._window_start <= idx < self._window_end):
                            self._load_window(idx)
                        if (
                            self._window_cache is not None
                            and layer_name in self._window_cache
                        ):
                            local_idx = idx - self._window_start
                            result[layer_name] = self._window_cache[layer_name][
                                local_idx
                            ]
                        else:
                            datasets = self._datasets
                            if datasets is None:
                                self._get_file_handle()
                                datasets = self._datasets
                            if layer_name in datasets:
                                embedding_data = datasets[layer_name][idx]
                                result[layer_name] = torch.from_numpy(
                                    np.array(embedding_data, dtype=np.float32)
                                )
                return result
            else:
                if "embed" in self._memory_cache:
                    return {
                        "embed": self._memory_cache["embed"][idx],
                        "label": self._memory_cache["labels"][idx],
                    }
                # Fallback to on-demand read if embeddings not cached
                if not (self._window_start <= idx < self._window_end):
                    self._load_window(idx)
                if self._window_cache is not None and "embed" in self._window_cache:
                    local_idx = idx - self._window_start
                    embed_tensor = self._window_cache["embed"][local_idx]
                else:
                    datasets = self._datasets
                    if datasets is None:
                        self._get_file_handle()
                        datasets = self._datasets
                    embed_data = datasets["embed"][idx]
                    embed_tensor = torch.from_numpy(
                        np.array(embed_data, dtype=np.float32)
                    )
                return {
                    "embed": embed_tensor,
                    "label": self._memory_cache["labels"][idx],
                }

        # Otherwise read from HDF5 (with proper worker handling)
        datasets = self._datasets
        if datasets is None:
            self._get_file_handle()
            datasets = self._datasets

        # Get label
        label_data = datasets["labels"][idx]
        label_tensor = torch.from_numpy(np.array(label_data, dtype=np.int64))

        if self.multi_layer:
            result = {"label": label_tensor}
            for layer_name in self.layer_names:
                if layer_name in datasets:
                    embedding_data = datasets[layer_name][idx]
                    result[layer_name] = torch.from_numpy(
                        np.array(embedding_data, dtype=np.float32)
                    )
            return result
        else:
            embed_data = datasets["embed"][idx]
            return {
                "embed": torch.from_numpy(np.array(embed_data, dtype=np.float32)),
                "label": label_tensor,
            }

    def get_embedding_dim(self, layer_name: str = None) -> Tuple[int, ...]:
        """Get embedding dimensions for a specific layer.

        Returns:
            Tuple[int, ...]: Embedding dimensions for the specified layer.
        """
        if layer_name is None:
            layer_name = self.layer_names[0]

        idx = self.layer_names.index(layer_name)
        return self.embedding_dims[idx]

    def close(self) -> None:
        """Explicitly close the HDF5 file."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass
            self._h5file = None
            self._datasets = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickling (needed for DataLoader workers).

        Returns:
            Dict[str, Any]: State dictionary for pickling.
        """
        state = self.__dict__.copy()
        # Remove file handle - will be reopened in worker
        state["_h5file"] = None
        state["_datasets"] = None
        state["_worker_id"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for unpickling."""
        self.__dict__.update(state)


# ----------------------------------------------------------------------------- #
# Utility: load / save pre-computed embeddings & labels to disk (HDF5)
# ----------------------------------------------------------------------------- #
def save_embeddings_arrays(
    embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    save_path: Path,
    num_labels: int,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Save already-computed embeddings/labels to an HDF5 file.

    Parameters
    ----------
    embeddings : Union[torch.Tensor, Dict[str, torch.Tensor]]
        Embeddings tensor (on CPU or GPU) or dictionary of layer embeddings.
        Will be moved to CPU and stored as float32.
    labels : torch.Tensor, shape (N,)
        Corresponding integer class labels (any integer dtype).
    save_path : Path
        Destination filepath. Parent directories are created automatically and
        file is overwritten if it already exists.
    num_labels : int
        Number of unique labels in the dataset.
    compression : str, optional
        HDF5 compression algorithm (default: "gzip").
    compression_level : int, optional
        Compression level for *gzip* (default: 4).
    """

    # Log embedding shapes before saving
    logger.info(f"Saving embeddings to {save_path}")
    if isinstance(embeddings, dict):
        for layer_name, layer_embeddings in embeddings.items():
            logger.info(f"  Layer '{layer_name}' shape: {layer_embeddings.shape}")
    else:
        logger.info(f"  Single tensor shape: {embeddings.shape}")

    # Ensure directory exists for local filesystem paths
    save_path_obj = anypath(save_path)
    if not (hasattr(save_path_obj, "is_cloud") and save_path_obj.is_cloud):
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Prepare numpy arrays - use explicit types for consistency
    # Embeddings: numpy.float32 for maximum precision and HDF5 compatibility
    # Labels: numpy.int64 for standard integer labels
    labels_np = labels.detach().cpu().numpy().astype(np.int64)

    # Write file – use file-like stream for cloud storage
    if hasattr(save_path_obj, "is_cloud") and save_path_obj.is_cloud:
        with save_path_obj.open("wb") as fh, h5py.File(fh, "w") as h5f:
            if isinstance(embeddings, dict):
                # Multi-layer embeddings: save each layer as a separate dataset
                layer_names = list(embeddings.keys())
                embedding_dims = [emb.shape[1:] for emb in embeddings.values()]

                for layer_name, layer_embeddings in embeddings.items():
                    layer_np = (
                        layer_embeddings.detach().cpu().numpy().astype(np.float32)
                    )
                    h5f.create_dataset(
                        f"embeddings_{layer_name}",
                        data=layer_np,
                        compression=compression,
                        compression_opts=compression_level,
                    )
                h5f.attrs["layer_names"] = layer_names
                # Convert embedding_dims to a format HDF5 can store
                # Store as a list of strings representing the dimensions
                h5f.attrs["embedding_dims"] = [
                    str(tuple(dim)) for dim in embedding_dims
                ]
                h5f.attrs["multi_layer"] = True
            else:
                # Single tensor: save as before for backward compatibility
                embeds_np = embeddings.detach().cpu().numpy().astype(np.float32)
                h5f.create_dataset(
                    "embeddings",
                    data=embeds_np,
                    compression=compression,
                    compression_opts=compression_level,
                )
                h5f.attrs["multi_layer"] = False
                # Store embedding dimensions for proper reshaping on load
                layer_names = ["embeddings"]  # Default layer name for single tensor
                embedding_dims = [embeds_np.shape[1:]]  # Exclude batch dimension
                h5f.attrs["layer_names"] = layer_names
                h5f.attrs["embedding_dims"] = [
                    str(tuple(dim)) for dim in embedding_dims
                ]

            h5f.create_dataset(
                "labels",
                data=labels_np,
                compression=compression,
                compression_opts=compression_level,
            )
            h5f.attrs["num_labels"] = num_labels
    else:
        with h5py.File(str(save_path_obj), "w") as h5f:
            if isinstance(embeddings, dict):
                # Multi-layer embeddings: save each layer as a separate dataset
                layer_names = list(embeddings.keys())
                embedding_dims = [emb.shape[1:] for emb in embeddings.values()]

                for layer_name, layer_embeddings in embeddings.items():
                    layer_np = (
                        layer_embeddings.detach().cpu().numpy().astype(np.float32)
                    )
                    h5f.create_dataset(
                        f"embeddings_{layer_name}",
                        data=layer_np,
                        compression=compression,
                        compression_opts=compression_level,
                    )
                h5f.attrs["layer_names"] = layer_names
                # Convert embedding_dims to a format HDF5 can store
                # Store as a list of strings representing the dimensions
                h5f.attrs["embedding_dims"] = [
                    str(tuple(dim)) for dim in embedding_dims
                ]
                h5f.attrs["multi_layer"] = True
            else:
                # Single tensor: save as before for backward compatibility
                embeds_np = embeddings.detach().cpu().numpy().astype(np.float32)
                h5f.create_dataset(
                    "embeddings",
                    data=embeds_np,
                    compression=compression,
                    compression_opts=compression_level,
                )
                h5f.attrs["multi_layer"] = False
                # Store embedding dimensions for proper reshaping on load
                layer_names = ["embeddings"]  # Default layer name for single tensor
                embedding_dims = [embeds_np.shape[1:]]  # Exclude batch dimension
                h5f.attrs["layer_names"] = layer_names
                h5f.attrs["embedding_dims"] = [
                    str(tuple(dim)) for dim in embedding_dims
                ]

            h5f.create_dataset(
                "labels",
                data=labels_np,
                compression=compression,
                compression_opts=compression_level,
            )
            h5f.attrs["num_labels"] = num_labels


def load_embeddings_arrays(
    path: Path,
) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor, int]:
    """Load embeddings & labels previously saved by *save_embeddings_arrays*.

    Parameters
    ----------
    path : Path
        Location of the *.h5* file

    Returns
    -------
    Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor, int]
        (embeddings, labels, num_labels) on CPU. embeddings is either a single tensor
        (for backward compatibility) or a dictionary with layer names as keys.

    Raises
    ------
    FileNotFoundError
        If the provided *path* does not exist.
    KeyError
        If required keys are missing from the HDF5 file.
    """

    # anypath handles both local and cloud paths
    path_obj = anypath(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    # Handle remote (cloud) paths by streaming through a file-like object
    if hasattr(path_obj, "is_cloud") and path_obj.is_cloud:
        with path_obj.open("rb") as fh, h5py.File(fh, "r") as h5f:
            labels = torch.from_numpy(np.asarray(h5f["labels"]))
            num_labels = h5f.attrs.get("num_labels", None)

            # Check if this is a multi-layer file
            # Prefer robust detection by inspecting dataset keys.
            has_prefixed = any(k.startswith("embeddings_") for k in h5f.keys())
            has_generic = "embeddings" in h5f
            is_multi_layer = has_prefixed or (
                h5f.attrs.get("multi_layer", False) and not has_generic
            )
            if is_multi_layer:
                # Load multi-layer embeddings
                layer_names = h5f.attrs.get("layer_names", [])
                embeds = {}
                for layer_name in layer_names:
                    embeds[layer_name] = torch.from_numpy(
                        np.asarray(h5f[f"embeddings_{layer_name}"], dtype=np.float32)
                    )
                    logger.info(
                        f"Loaded embeddings for layer '{layer_name}' shape: "
                        f"{embeds[layer_name].shape}"
                    )
            else:
                # Load single tensor for backward compatibility. If only a prefixed
                # dataset exists (older streaming), fallback to it.
                if "embeddings" in h5f:
                    embeds = torch.from_numpy(
                        np.asarray(h5f["embeddings"], dtype=np.float32)
                    )
                    logger.info(
                        f"Loaded single tensor embeddings shape: {embeds.shape}"
                    )
                else:
                    # Fallback: use the first prefixed dataset
                    first_key = next(
                        (k for k in h5f.keys() if k.startswith("embeddings_")), None
                    )
                    if first_key is None:
                        raise KeyError("No embeddings dataset found in HDF5 file")
                    embeds = torch.from_numpy(
                        np.asarray(h5f[first_key], dtype=np.float32)
                    )
                    logger.info(
                        f"Loaded fallback embeddings from '{first_key}' shape: "
                        f"{embeds.shape}"
                    )
    else:
        with h5py.File(str(path_obj), "r") as h5f:
            labels = torch.from_numpy(np.asarray(h5f["labels"]))
            num_labels = h5f.attrs.get("num_labels", None)

            # Check if this is a multi-layer file (robust detection)
            has_prefixed = any(k.startswith("embeddings_") for k in h5f.keys())
            has_generic = "embeddings" in h5f
            is_multi_layer = has_prefixed or (
                h5f.attrs.get("multi_layer", False) and not has_generic
            )
            if is_multi_layer:
                # Load multi-layer embeddings from separate datasets
                layer_names = h5f.attrs.get("layer_names", [])
                embeds = {}
                for layer_name in layer_names:
                    layer_embeds = torch.from_numpy(
                        np.asarray(h5f[f"embeddings_{layer_name}"], dtype=np.float32)
                    )
                    logger.info(
                        f"Loaded embeddings for layer '{layer_name}' shape: "
                        f"{layer_embeds.shape}"
                    )
                    embeds[layer_name] = layer_embeds
            else:
                # Load single tensor for backward compatibility, with fallback
                if "embeddings" in h5f:
                    embeds = torch.from_numpy(
                        np.asarray(h5f["embeddings"], dtype=np.float32)
                    )
                    logger.info(
                        f"Loaded single tensor embeddings shape: {embeds.shape}"
                    )
                else:
                    first_key = next(
                        (k for k in h5f.keys() if k.startswith("embeddings_")), None
                    )
                    if first_key is None:
                        raise KeyError("No embeddings dataset found in HDF5 file")
                    embeds = torch.from_numpy(
                        np.asarray(h5f[first_key], dtype=np.float32)
                    )
                    logger.info(
                        f"Loaded fallback embeddings from '{first_key}' shape: "
                        f"{embeds.shape}"
                    )

    return embeds, labels, num_labels
