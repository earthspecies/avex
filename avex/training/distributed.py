"""Distributed training setup utilities."""

import datetime as _dt
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_slurm_env() -> Tuple[int, int, int, int, str]:
    """
    Get SLURM environment variables for distributed training.

    Returns
    -------
    Tuple[int, int, int, int, str]
        node_id, local_rank, global_rank, world_size, master_addr
    """
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    master_addr = os.environ.get("SLURM_NODELIST").split(",")[0]
    return node_id, local_rank, global_rank, world_size, master_addr


def get_local_device_index() -> int:
    """
    Get the local CUDA device index for this process.

    In SLURM environments, this uses SLURM_LOCALID but ensures the device
    index is valid. Falls back to auto-detection if needed.

    Returns
    -------
    int
        The CUDA device index this process should use
    """
    import torch

    if not torch.cuda.is_available():
        return 0

    visible_devices = torch.cuda.device_count()

    # In SLURM, if only one GPU is visible per process, use device 0
    if visible_devices == 1:
        return 0

    # For multi-GPU setups, use SLURM_LOCALID if available
    if "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        # Ensure the device index is valid
        if local_rank < visible_devices:
            return local_rank
        else:
            logger.warning(f"SLURM_LOCALID={local_rank} >= visible devices {visible_devices}. Using device 0.")
            return 0

    # Fallback: use LOCAL_RANK from torchrun if available
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        return min(local_rank, visible_devices - 1)

    # Final fallback
    return 0


def init_distributed(port: int = 29500, backend: str = "nccl") -> Tuple[int, int, bool]:
    """
    Initialize distributed training.

    This function sets up the distributed training environment based on SLURM
    environment variables. If SLURM variables are not found, it attempts to
    initialize with `torch.distributed.init_process_group`.

    Parameters
    ----------
    port : int, optional
        Port number for the master node, by default 29500
    backend : str, optional
        Distributed backend to use, by default "nccl"

    Returns
    -------
    Tuple[int, int, bool]
        Tuple of (rank, world_size, is_distributed)
    """
    is_distributed = False
    rank = 0
    world_size = 1

    # Check for backend override from environment variable
    env_backend = os.environ.get("PYTORCH_DISTRIBUTED_BACKEND", backend)
    if env_backend != backend:
        logger.info(f"Using backend '{env_backend}' from PYTORCH_DISTRIBUTED_BACKEND environment variable")
        backend = env_backend

    # Check if SLURM environment variables are set
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        node_id, local_rank, global_rank, world_size, master_addr = get_slurm_env()
        logger.info(
            "SLURM env variables: node_id=%s, local_rank=%s, global_rank=%s, world_size=%s, master_addr=%s",
            node_id,
            local_rank,
            global_rank,
            world_size,
            master_addr,
        )

        # Only initialize distributed training if we have multiple GPUs/tasks
        if world_size > 1:
            rank = global_rank
            os.environ["MASTER_ADDR"] = master_addr
            # Use a unique port for each job to avoid conflicts
            job_id_str = os.environ.get("SLURM_JOB_ID", "0")
            try:
                job_id = int(job_id_str)
                job_port = port + job_id % 1000
            except ValueError:
                logger.warning(
                    "Could not parse SLURM_JOB_ID: %s. Using default port.",
                    job_id_str,
                )
                job_port = port
            os.environ["MASTER_PORT"] = str(job_port)

            # Get the correct local device index
            try:
                import torch

                if torch.cuda.is_available():
                    local_device_index = get_local_device_index()
                    torch.cuda.set_device(local_device_index)
                    logger.info(
                        f"Set CUDA device to {local_device_index} "
                        f"(SLURM_LOCALID={local_rank}, "
                        f"visible_devices={torch.cuda.device_count()})"
                    )
            except Exception as e:  # pragma: no cover – log but don't crash
                logger.warning("Failed to set CUDA device: %s", e)

            logger.info(
                f"Initializing distributed training with rank {rank}, "
                f"world size {world_size}, master addr {master_addr}:{job_port}"
            )
            dist.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=_dt.timedelta(minutes=5),
            )
            is_distributed = True
            logger.info("Distributed training initialized successfully.")

        else:
            logger.info("Single GPU/task detected (world_size=1). Skipping distributed initialization.")

    elif dist.is_available() and not dist.is_initialized():
        # Fallback for non-SLURM environments if needed, e.g. torchrun
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size > 1:
                # Ensure each process uses the correct local GPU when launched
                # via `torchrun`/`torch.distributed.run` (LOCAL_RANK is set).
                try:
                    import torch

                    local_gpu_idx = get_local_device_index()
                    if torch.cuda.is_available():
                        torch.cuda.set_device(local_gpu_idx)
                        logger.info("Set CUDA device to LOCAL_RANK=%s", local_gpu_idx)
                except Exception as e:  # pragma: no cover – best-effort only
                    logger.warning("Failed to set CUDA device: %s", e)

                dist.init_process_group(backend=backend, init_method="env://")
                is_distributed = True
                logger.info(f"Initialized torch.distributed via env:// (rank {rank}, world_size {world_size})")
        else:
            logger.info("Neither SLURM nor standard torch.distributed env vars found. Running in non-distributed mode.")

    elif dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = world_size > 1
        logger.info(
            "torch.distributed already initialized "
            f"(rank {rank}, world_size {world_size}, is_distributed={is_distributed})"
        )
    else:
        logger.info("torch.distributed not available. Running in non-distributed mode.")

    return rank, world_size, is_distributed


def get_rank() -> int:
    """Returns the rank of the current process, or 0 if not distributed.

    Returns
    -------
    int
        The rank of the current process if distributed, otherwise 0.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Returns the world size, or 1 if not distributed.

    Returns
    -------
    int
        The world size if distributed, otherwise 1.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def cleanup_distributed() -> None:
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0).

    Returns
    -------
    bool
        True if the current process is rank 0 or if not distributed, False otherwise.
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True  # If not initialized, assume single process


def synchronize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Synchronize a scalar tensor across all ranks by summing and averaging.

    Returns
    -------
    torch.Tensor
        The synchronized tensor averaged across all ranks
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    # Ensure tensor is on the correct device and is a scalar
    if tensor.dim() != 0:
        tensor = tensor.sum()

    # All-reduce to sum across ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Average by world size
    world_size = dist.get_world_size()
    tensor = tensor / world_size

    return tensor


def synchronize_scalar(value: float, device: torch.device) -> float:
    """Synchronize a scalar value across all ranks.

    Returns
    -------
    float
        The synchronized scalar value averaged across all ranks
    """
    import torch

    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    synchronized_tensor = synchronize_tensor(tensor)
    return synchronized_tensor.item()


def gather_metrics_from_all_ranks(
    total_loss: float,
    total_correct: int,
    total_samples: int,
    device: torch.device,
    is_clip_mode: bool = False,
    total_correct_a2t: int = 0,
    total_correct_t2a: int = 0,
) -> Tuple[float, float]:
    """Gather and synchronize metrics from all ranks.

    Returns
    -------
    Tuple[float, float]
        A tuple of (average_loss, average_accuracy) synchronized across all ranks
    """
    if not dist.is_available() or not dist.is_initialized():
        if is_clip_mode:
            avg_acc = (total_correct_a2t + total_correct_t2a) / 2.0 / total_samples if total_samples > 0 else 0.0
        else:
            avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
        return (total_loss / total_samples if total_samples > 0 else 0.0), avg_acc

    # Synchronize metrics across ranks
    total_loss_sync = synchronize_scalar(total_loss, device)
    total_samples_sync = synchronize_scalar(total_samples, device)

    if is_clip_mode:
        total_correct_a2t_sync = synchronize_scalar(total_correct_a2t, device)
        total_correct_t2a_sync = synchronize_scalar(total_correct_t2a, device)
        avg_acc = (
            (total_correct_a2t_sync + total_correct_t2a_sync) / 2.0 / total_samples_sync
            if total_samples_sync > 0
            else 0.0
        )
    else:
        total_correct_sync = synchronize_scalar(total_correct, device)
        avg_acc = total_correct_sync / total_samples_sync if total_samples_sync > 0 else 0.0

    avg_loss = total_loss_sync / total_samples_sync if total_samples_sync > 0 else 0.0

    return avg_loss, avg_acc
