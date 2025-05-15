"""Distributed training setup utilities."""

import builtins
import logging
import os
from typing import Tuple

import torch.distributed as dist

logger = logging.getLogger(__name__)


# Disable printing when not in master process
def suppress_non_master_prints(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args: object, **kwargs: object) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


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
    # master_addr = os.environ.get(
    #     "SLURM_LAUNCH_NODE_IPADDR", socket.gethostbyname(socket.gethostname())
    # )
    master_addr = os.environ.get("SLURM_NODELIST").split(",")[0]
    return node_id, local_rank, global_rank, world_size, master_addr


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

    # Check if SLURM environment variables are set
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        node_id, local_rank, global_rank, world_size, master_addr = get_slurm_env()
        logger.info(
            "SLURM env variables: node_id=%s, local_rank=%s, global_rank=%s, "
            "world_size=%s, master_addr=%s",
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
                    "Could not parse SLURM_JOB_ID: %s. Using default port.", job_id_str
                )
                job_port = port
            os.environ["MASTER_PORT"] = str(job_port)

            logger.info(
                f"Initializing distributed training with rank {rank}, "
                f"world size {world_size}, master addr {master_addr}:{job_port}"
            )
            dist.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
            )
            suppress_non_master_prints(rank == 0)
            is_distributed = True
            logger.info("Distributed training initialized successfully.")
        else:
            logger.info(
                "Single GPU/task detected (world_size=1). "
                "Skipping distributed initialization."
            )
            suppress_non_master_prints(True)  # Standalone process is master

    elif dist.is_available() and not dist.is_initialized():
        # Fallback for non-SLURM environments if needed, e.g. torchrun
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size > 1:
                dist.init_process_group(backend=backend, init_method="env://")
                suppress_non_master_prints(rank == 0)
                is_distributed = True
                logger.info(
                    f"Initialized torch.distributed via env:// "
                    f"(rank {rank}, world_size {world_size})"
                )
            else:
                suppress_non_master_prints(True)
        else:
            logger.info(
                "Neither SLURM nor standard torch.distributed env vars found. "
                "Running in non-distributed mode."
            )
            suppress_non_master_prints(True)  # Standalone process is master
    elif dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = world_size > 1
        suppress_non_master_prints(rank == 0)
        logger.info(
            "torch.distributed already initialized "
            f"(rank {rank}, world_size {world_size}, is_distributed={is_distributed})"
        )
    else:
        logger.info("torch.distributed not available. Running in non-distributed mode.")
        suppress_non_master_prints(True)  # Standalone process is master

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
