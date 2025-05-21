from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)

# Optional cloud storage support (gs://)
try:
    from cloudpathlib import GSPath  # type: ignore
except ImportError:  # pragma: no cover – cloudpathlib optional
    GSPath = None  # type: ignore


def extract_embeddings_for_split(
    model: ModelBase,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return stacked embeddings and labels for an entire dataloader.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (embeddings, labels) stacked on CPU.
    """
    embeds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            wav = batch["raw_wav"].to(device)
            mask = batch.get("padding_mask")
            if mask is not None:
                mask = mask.to(device)
            if mask is None:
                emb = model.extract_embeddings(wav, layer_names)
            else:
                inp = {"raw_wav": wav, "padding_mask": mask}
                emb = model.extract_embeddings(inp, layer_names)
            embeds.append(emb.cpu())
            labels.append(batch["label"].cpu())

    return torch.cat(embeds), torch.cat(labels)


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
) -> None:
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

    with h5py.File(h5_path, "w", libver="latest") as h5f:
        # Get first batch to determine shapes
        first_batch = next(iter(dataloader))
        sample_embeddings = model.extract_embeddings(first_batch[0], layer_names)
        embedding_dim = sample_embeddings.shape[1]
        total_samples = len(dataloader.dataset)

        # Calculate optimal chunk size based on available memory
        available_memory = (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else 8 * 1024 * 1024 * 1024
        )  # 8GB default
        chunk_size = min(
            100, max(1, int(available_memory / (embedding_dim * 4 * 1024 * 1024)))
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

                embeddings = model.extract_embeddings(x, layer_names)
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


def load_embeddings_arrays(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings & labels previously saved by *save_embeddings_arrays*.

    Parameters
    ----------
    path : Path
        Location of the *.h5* file

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (embeddings, labels) on CPU

    Raises
    ------
    FileNotFoundError
        If the provided *path* does not exist.
    """

    # cloudpathlib.GSPath and pathlib.Path both provide .exists and .open
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    # Handle remote (gs://) paths by streaming through a file-like object
    if GSPath is not None and isinstance(path, GSPath):
        with path.open("rb") as fh, h5py.File(fh, "r") as h5f:
            embeds = torch.from_numpy(np.asarray(h5f["embeddings"], dtype=np.float32))
            labels = torch.from_numpy(np.asarray(h5f["labels"]))
    else:
        with h5py.File(str(path), "r") as h5f:
            embeds = torch.from_numpy(np.asarray(h5f["embeddings"], dtype=np.float32))
            labels = torch.from_numpy(np.asarray(h5f["labels"]))

    return embeds, labels


# ----------------------------------------------------------------------------- #
# Save routine
# ----------------------------------------------------------------------------- #


def save_embeddings_arrays(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
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
    compression : str, optional
        HDF5 compression algorithm (default: "gzip").
    compression_level : int, optional
        Compression level for *gzip* (default: 4).
    """

    # Ensure directory exists for local filesystem paths
    if not (GSPath is not None and isinstance(save_path, GSPath)):
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare numpy arrays
    embeds_np = embeddings.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy()

    # Write file – use file-like stream for GCS
    if GSPath is not None and isinstance(save_path, GSPath):
        with save_path.open("wb") as fh, h5py.File(fh, "w") as h5f:
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
    else:
        with h5py.File(str(save_path), "w") as h5f:
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
