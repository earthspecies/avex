import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
