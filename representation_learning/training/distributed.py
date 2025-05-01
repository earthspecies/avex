"""Distributed training setup utilities."""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import logging
import subprocess

logger = logging.getLogger("run_train")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_slurm_available() -> bool:
    """Check if running in a Slurm environment.

    Returns
    -------
    bool
        True if running in Slurm, False otherwise
    """
    return "SLURM_JOB_ID" in os.environ


def _first_host(nodelist_var: str = "SLURM_JOB_NODELIST") -> str:
    nodelist = os.environ[nodelist_var]
    return subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist], text=True
    ).splitlines()[0]

def get_slurm_env() -> Tuple[int, int, int, int, str]:
    if not is_slurm_available():
        return 0, 0, 0, 1, "localhost"

    node_id     = int(os.environ["SLURM_NODEID"])
    local_rank  = int(os.environ["SLURM_LOCALID"])
    global_rank = int(os.environ["SLURM_PROCID"])
    world_size  = int(os.environ["SLURM_NTASKS"])
    master_addr = _first_host()          # JOB_NODELIST by default

    return node_id, local_rank, global_rank, world_size, master_addr



def setup_distributed(
    backend: str = "nccl",
    port: int = 29500,
) -> Tuple[Optional[int], int, str]:
    """Setup distributed training environment.

    Parameters
    ----------
    backend : str, optional
        Distributed backend to use, by default "nccl"
    port : int, optional
        Base port for distributed training, by default 29500

    Returns
    -------
    Tuple[Optional[int], int, str]
        (local_rank, world_size, master_addr)
        If not in distributed mode, local_rank will be None
    """
    if not is_slurm_available():
        return None, 1, "localhost"

    # Get Slurm environment variables
    node_id, local_rank, global_rank, world_size, master_addr = get_slurm_env()
    logger.info("slurm env variables: %s", get_slurm_env())

    # Only initialize distributed training if we have multiple GPUs
    if world_size <= 1:
        return None, 1, master_addr

    # Calculate the port for this job to avoid conflicts
    job_port = port + int(os.environ["SLURM_JOB_ID"]) % 1000

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{job_port}",
            world_size=world_size,
            rank=global_rank
        )
    
    torch.distributed.barrier()
    setup_for_distributed(local_rank == 0)

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return local_rank, world_size, master_addr


def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master_process() -> bool:
    """Check if current process is the master process.

    Returns
    -------
    bool
        True if this is the master process, False otherwise
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the total number of processes.

    Returns
    -------
    int
        Total number of processes
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process.

    Returns
    -------
    int
        Rank of the current process
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def all_gather_object(obj: any) -> list:
    """Gather objects from all processes.

    Parameters
    ----------
    obj : any
        Object to gather

    Returns
    -------
    list
        List of objects from all processes
    """
    if not dist.is_initialized():
        return [obj]

    world_size = dist.get_world_size()
    gathered_objects = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary across all processes.

    Parameters
    ----------
    input_dict : dict
        Dictionary to reduce
    average : bool, optional
        Whether to average the values, by default True

    Returns
    -------
    dict
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict

    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k, v in sorted(input_dict.items()):
            names.append(k)
            values.append(v)

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict 