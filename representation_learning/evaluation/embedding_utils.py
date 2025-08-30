from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from esp_data.io import anypath
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


def extract_embeddings_for_split(
    model: ModelBase,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    *,
    aggregation: str = "mean",
    save_path: Optional[Path] = None,
    chunk_size: int = 1000,
    compression: str = "gzip",
    compression_level: int = 4,
    auto_chunk_size: bool = True,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100,
    batch_chunk_size: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return stacked embeddings and labels for an entire dataloader.

    This function uses a memory-efficient streaming approach to avoid OOM issues
    when extracting embeddings from many layers (e.g., "all" layers in EfficientNet).

    Parameters
    ----------
    model : ModelBase
        The model to extract embeddings from
    dataloader : DataLoader
        DataLoader containing the samples
    layer_names : List[str]
        List of layer names to extract embeddings from
    device : torch.device
        Device to run the model on
    aggregation : str, optional
        Aggregation method for embeddings: "mean", "max", "cls_token", or "none".
        Default is "mean".
    save_path : Optional[Path], optional
        If provided, embeddings are saved directly to disk in chunks to save memory.
        This is especially useful when extracting from many layers.
    chunk_size : int, optional
        Number of samples to process in memory at once when saving to disk.
        Default is 1000.
    compression : str, optional
        HDF5 compression algorithm when saving to disk. Default is "gzip".
    compression_level : int, optional
        Compression level for gzip when saving to disk. Default is 4.
    auto_chunk_size : bool, optional
        If True, automatically calculate optimal chunk size based on available GPU
        memory.
        If False, use chunk_size directly. Default is True.
    max_chunk_size : int, optional
        Maximum chunk size when auto-calculating. Default is 2000.
    min_chunk_size : int, optional
        Minimum chunk size when auto-calculating. Default is 100.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (embeddings, labels) stacked on CPU.
    """
    model.eval()

    # If save_path is provided, use streaming approach to save memory
    if save_path is not None:
        return _extract_embeddings_streaming(
            model,
            dataloader,
            layer_names,
            device,
            save_path,
            chunk_size,
            compression,
            compression_level,
            aggregation,
            auto_chunk_size,
            max_chunk_size,
            min_chunk_size,
            batch_chunk_size,
        )

    # Original in-memory approach for backward compatibility
    return _extract_embeddings_in_memory(
        model, dataloader, layer_names, device, aggregation
    )


def _extract_embeddings_in_memory(
    model: ModelBase,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    aggregation: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Original in-memory embedding extraction (kept for backward compatibility).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (embeddings, labels) stacked on CPU.
    """

    embeds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    progress = tqdm(
        enumerate(dataloader),
        desc="Extracting embeddings (in-memory)",
        total=len(dataloader),
        unit="batch",
    )

    logger.info(f"Extracting embeddings for {len(dataloader)} batches (in-memory mode)")

    try:
        with torch.no_grad():
            # Register hooks for the specified layers outside the loop
            model.register_hooks_for_layers(layer_names)

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

                embeds.append(emb.cpu())
                labels.append(batch["label"].cpu())

        logger.info(f"Extracted {len(embeds)} embeddings")

        return torch.cat(embeds), torch.cat(labels)

    finally:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient streaming embedding extraction that saves to disk in chunks.

    This hybrid approach processes multiple batches in memory before writing to disk,
    significantly reducing I/O overhead while maintaining memory efficiency.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (embeddings, labels) loaded from disk.

    Raises
    ------
    ValueError
        If invalid parameters are provided.
    """
    # Register hooks for the specified layers outside the loop
    model.register_hooks_for_layers(layer_names)

    logger.info(
        f"Extracting embeddings using optimized streaming approach to {save_path}"
    )
    logger.info(f"Chunk size: {chunk_size}, layers: {len(layer_names)}")

    # Ensure directory exists
    save_path_obj = anypath(save_path)
    if not save_path_obj.is_cloud:
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

    embedding_dim = sample_emb.shape[1]
    total_samples = len(dataloader.dataset)

    logger.info(f"Embedding dimension: {embedding_dim}, total samples: {total_samples}")

    # Calculate optimal chunk size if auto-calculation is enabled
    if auto_chunk_size and torch.cuda.is_available():
        try:
            available_memory = torch.cuda.get_device_properties(0).total_memory
            # Calculate optimal chunk size based on available memory
            # Estimate memory per sample: embedding_dim * 4 bytes (float32) + overhead
            memory_per_sample = embedding_dim * 4 + 1024  # Add 1KB overhead per sample
            optimal_chunk_size = int(
                available_memory * 0.15 / memory_per_sample
            )  # Use 15% of GPU memory

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
    # HDF5 has a 4GB chunk size limit: chunk_size * embedding_dim * 4 bytes < 4GB
    max_chunk_size_for_hdf5 = int(
        4 * 1024 * 1024 * 1024 / (embedding_dim * 4)
    )  # 4GB limit
    if chunk_size > max_chunk_size_for_hdf5:
        logger.warning(
            f"Chunk size {chunk_size} exceeds HDF5 4GB limit for "
            f"embedding_dim {embedding_dim}. Adjusting to {max_chunk_size_for_hdf5}."
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
        if save_path_obj.is_cloud:
            with save_path_obj.open("wb") as fh, h5py.File(fh, "w") as h5f:
                _create_and_fill_h5_datasets_hybrid(
                    h5f,
                    total_samples,
                    embedding_dim,
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
                )
        else:
            with h5py.File(str(save_path_obj), "w") as h5f:
                _create_and_fill_h5_datasets_hybrid(
                    h5f,
                    total_samples,
                    embedding_dim,
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
                )

        # Load the saved embeddings back into memory
        logger.info("Loading saved embeddings back into memory")
        embeddings, labels, _ = load_embeddings_arrays(save_path)
        return embeddings, labels

    finally:
        # Always deregister hooks when done
        model.deregister_all_hooks()


def _create_and_fill_h5_datasets_hybrid(
    h5f: h5py.File,
    total_samples: int,
    embedding_dim: int,
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
) -> None:
    """Helper function to create and fill HDF5 datasets with optimized hybrid approach.

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

    # Create resizable datasets
    embeddings_dset = h5f.create_dataset(
        "embeddings",
        shape=(total_samples, embedding_dim),
        maxshape=(None, embedding_dim),
        dtype=np.float32,
        chunks=(chunk_size, embedding_dim),
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

    logger.info(
        f"Dataset shapes - embeddings: {embeddings_dset.shape}, "
        f"labels: {labels_dset.shape}"
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

        batch_size = len(embeddings)

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
            # Ensure we're working with numpy arrays
            embeddings_np = embeddings.cpu().numpy()
            labels_np = first_batch_labels.numpy()

            embeddings_dset[start_idx : start_idx + batch_size] = embeddings_np
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
        progress = tqdm(
            enumerate(dataloader),
            desc="Extracting embeddings (hybrid streaming)",
            total=len(dataloader),
            unit="batch",
        )

        # Buffers for collecting multiple batches before writing
        batch_embeddings_buffer = []
        batch_labels_buffer = []
        buffer_sample_count = 0

        for batch_idx, batch in progress:
            wav = batch["raw_wav"].to(device)
            mask = batch.get("padding_mask")
            if mask is not None:
                mask = mask.to(device)

            if mask is None:
                embeddings = model.extract_embeddings(wav, aggregation=aggregation)
            else:
                inp = {"raw_wav": wav, "padding_mask": mask}
                embeddings = model.extract_embeddings(inp, aggregation=aggregation)

            batch_size = len(embeddings)

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

            # Add to buffer
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
                    # Concatenate embeddings
                    if len(batch_embeddings_buffer) == 1:
                        concatenated_embeddings = batch_embeddings_buffer[0]
                    else:
                        concatenated_embeddings = torch.cat(
                            batch_embeddings_buffer, dim=0
                        )

                    # Concatenate labels
                    if len(batch_labels_buffer) == 1:
                        concatenated_labels = batch_labels_buffer[0]
                    else:
                        concatenated_labels = torch.cat(batch_labels_buffer, dim=0)

                    # Convert to numpy
                    embeddings_np = concatenated_embeddings.numpy()
                    labels_np = concatenated_labels.numpy()

                    # Validate we won't exceed dataset bounds
                    if start_idx + len(embeddings_np) > total_samples:
                        logger.warning(
                            "Buffer would exceed dataset bounds. Adjusting buffer size."
                        )
                        remaining_samples = total_samples - start_idx
                        if remaining_samples > 0:
                            embeddings_np = embeddings_np[:remaining_samples]
                            labels_np = labels_np[:remaining_samples]
                        else:
                            break

                    # Write to HDF5
                    embeddings_dset[start_idx : start_idx + len(embeddings_np)] = (
                        embeddings_np
                    )
                    labels_dset[start_idx : start_idx + len(labels_np)] = labels_np

                    start_idx += len(embeddings_np)

                    logger.debug(
                        f"Wrote buffer to disk - samples: {len(embeddings_np)}, "
                        f"total processed: {start_idx}/{total_samples}"
                    )

                except Exception as e:
                    logger.error(f"Error writing buffer to HDF5: {e}")
                    raise

                # Clear buffers
                batch_embeddings_buffer.clear()
                batch_labels_buffer.clear()
                buffer_sample_count = 0

                # Clear memory
                del concatenated_embeddings, concatenated_labels
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
    logger.info(
        f"Final dataset shapes - embeddings: {embeddings_dset.shape}, "
        f"labels: {labels_dset.shape}"
    )


class EmbeddingDataset(torch.utils.data.Dataset):
    """Simple dataset that serves pre-computed embeddings and labels."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:  # noqa: D401
        return self.embeddings.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: D401
        return {"embed": self.embeddings[idx], "label": self.labels[idx]}


def save_embeddings_to_disk(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_names: List[str],
    save_dir: Path,
    split: str,
    aggregation: str = "mean",
) -> None:
    """
    Save embeddings for all samples in a dataloader to disk using HDF5 format.

    Args:
        model: The model to extract embeddings from
        dataloader: DataLoader containing the samples
        layer_names: List of layer names to extract embeddings from
        save_dir: Directory to save embeddings
        split: Dataset split name (e.g., 'train' or 'val')
        aggregation: Aggregation method for embeddings ('mean', 'max', 'cls_token',
            'none')
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

    # Register hooks for the specified layers outside the loop
    if hasattr(model, "register_hooks_for_layers"):
        model.register_hooks_for_layers(layer_names)

    try:
        with h5py.File(h5_path, "w", libver="latest") as h5f:
            # Get first batch to determine shapes
            first_batch = next(iter(dataloader))
            sample_embeddings = model.extract_embeddings(
                first_batch[0], aggregation=aggregation
            )
            embedding_dim = sample_embeddings.shape[1]
            total_samples = len(dataloader.dataset)

            # Calculate optimal chunk size based on available memory
            available_memory = (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else 8 * 1024 * 1024 * 1024
            )  # 8GB default
            chunk_size = min(
                100,
                max(1, int(available_memory / (embedding_dim * 4 * 1024 * 1024))),
            )  # 4 bytes per float32

            logger.info(f"Embedding dimension: {embedding_dim}")
            logger.info(f"Chunk size: {chunk_size}")

            # Create resizable datasets with compression
            embeddings_dset = h5f.create_dataset(
                "embeddings",
                shape=(total_samples, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
                chunks=(chunk_size, embedding_dim),
                compression="gzip",
                compression_opts=4,
            )
            labels_dset = h5f.create_dataset(
                "labels",
                shape=(total_samples,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(chunk_size,),
                compression="gzip",
                compression_opts=4,
            )

            # Save embeddings and labels in chunks
            start_idx = 0
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(
                    tqdm(dataloader, desc=f"Saving {split} embeddings")
                ):
                    # Log progress every 10 batches
                    if batch_idx % 10 == 0:
                        logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")

                    embeddings = model.extract_embeddings(x, aggregation=aggregation)
                    batch_size = len(embeddings)
                    # Write to HDF5
                    embeddings_dset[start_idx : start_idx + batch_size] = (
                        embeddings.cpu().numpy()
                    )
                    labels_dset[start_idx : start_idx + batch_size] = y.numpy()

                    start_idx += batch_size

                    # Clear memory
                    del embeddings
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    # Force flush to disk periodically
                    if batch_idx % 100 == 0:
                        h5f.flush()

    finally:
        # Always deregister hooks when done
        if hasattr(model, "deregister_all_hooks"):
            model.deregister_all_hooks()


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
        def __init__(self, h5_path: Path) -> None:
            self.h5_path = h5_path
            self.h5_file = h5py.File(h5_path, "r")
            self.embeddings = self.h5_file["embeddings"]
            self.labels = self.h5_file["labels"]

        def __len__(self) -> int:
            return len(self.embeddings)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            # HDF5 handles efficient reading of individual samples
            return torch.from_numpy(self.embeddings[idx]), torch.from_numpy(
                self.labels[idx]
            )

        def __del__(self) -> None:
            # Ensure HDF5 file is closed when dataset is deleted
            if hasattr(self, "h5_file"):
                self.h5_file.close()

    return HDF5EmbeddingDataset(save_dir / "embeddings" / f"{split}.h5")


# ----------------------------------------------------------------------------- #
# Utility: load / save pre-computed embeddings & labels to disk (HDF5)
# ----------------------------------------------------------------------------- #


def save_embeddings_arrays(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    num_labels: int,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Save already-computed embeddings/labels to an HDF5 file.

    Parameters
    ----------
    embeddings : torch.Tensor, shape (N, D)
        Embeddings tensor (on CPU or GPU). Will be moved to CPU and stored as
        float32.
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

    # Ensure directory exists for local filesystem paths
    save_path_obj = anypath(save_path)
    if not save_path_obj.is_cloud:
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Prepare numpy arrays
    embeds_np = embeddings.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy()

    # Write file – use file-like stream for cloud storage
    if save_path_obj.is_cloud:
        with save_path_obj.open("wb") as fh, h5py.File(fh, "w") as h5f:
            h5f.create_dataset(
                "embeddings",
                data=embeds_np,
                compression=compression,
                compression_opts=compression_level,
            )
            h5f.create_dataset(
                "labels",
                data=labels_np,
                compression=compression,
                compression_opts=compression_level,
            )
            h5f.attrs["num_labels"] = num_labels
    else:
        with h5py.File(str(save_path_obj), "w") as h5f:
            h5f.create_dataset(
                "embeddings",
                data=embeds_np,
                compression=compression,
                compression_opts=compression_level,
            )
            h5f.create_dataset(
                "labels",
                data=labels_np,
                compression=compression,
                compression_opts=compression_level,
            )
            h5f.attrs["num_labels"] = num_labels


def load_embeddings_arrays(
    path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load embeddings & labels previously saved by *save_embeddings_arrays*.

    Parameters
    ----------
    path : Path
        Location of the *.h5* file

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, int]
        (embeddings, labels, num_labels) on CPU

    Raises
    ------
    FileNotFoundError
        If the provided *path* does not exist.
    """

    # anypath handles both local and cloud paths
    path_obj = anypath(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    # Handle remote (cloud) paths by streaming through a file-like object
    if path_obj.is_cloud:
        with path_obj.open("rb") as fh, h5py.File(fh, "r") as h5f:
            embeds = torch.from_numpy(np.asarray(h5f["embeddings"], dtype=np.float32))
            labels = torch.from_numpy(np.asarray(h5f["labels"]))
            num_labels = h5f.attrs.get("num_labels", None)
    else:
        with h5py.File(str(path_obj), "r") as h5f:
            embeds = torch.from_numpy(np.asarray(h5f["embeddings"], dtype=np.float32))
            labels = torch.from_numpy(np.asarray(h5f["labels"]))
            num_labels = h5f.attrs.get("num_labels", None)

    return embeds, labels, num_labels
