"""
Factory for creating trainers based on configuration.

This module provides a factory for creating properly configured trainers
based on the run configuration and training parameters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from representation_learning.configs import RunConfig
from representation_learning.training.checkpoint_manager import (
    CheckpointManager,
)

# Remove distributed coordinator - using functions directly from distributed.py
from representation_learning.training.losses import build_criterion
from representation_learning.training.metrics_tracker import MetricsTracker
from representation_learning.training.training_strategies import (
    StrategyFactory,
)

# Import will be done in the method to avoid circular imports
from representation_learning.utils import ExperimentLogger

if TYPE_CHECKING:
    from representation_learning.training.train import Trainer

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory for creating trainers."""

    @staticmethod
    def create_trainer(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: RunConfig,
        local_rank: int,
        world_size: int,
        is_distributed: bool,
        device: torch.device,
        exp_logger: Optional[ExperimentLogger] = None,
        num_classes: int = 1000,
        resume_from_checkpoint: Optional[str] = None,
    ) -> "Trainer":
        """Create a trainer based on configuration.

        Parameters
        ----------
        model : nn.Module
            Model to train
        optimizer : torch.optim.Optimizer
            Optimizer instance
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
            Learning rate scheduler
        scaler : Optional[torch.cuda.amp.GradScaler]
            AMP scaler for mixed precision training
        train_dataloader : DataLoader
            Training data loader
        eval_dataloader : DataLoader
            Evaluation data loader
        config : RunConfig
            Training configuration
        local_rank : int
            Local rank for distributed training
        world_size : int
            World size for distributed training
        is_distributed : bool
            Whether distributed training is enabled
        device : torch.device
            Device for training
        exp_logger : Optional[ExperimentLogger], optional
            Experiment logger, by default None
        num_classes : int, optional
            Number of classes, by default 1000
        resume_from_checkpoint : Optional[str], optional
            Path to checkpoint to resume from, by default None

        Returns
        -------
        ModularTrainer
            Configured trainer instance

        Raises
        ------
        ValueError
            If clustering evaluation is configured but model doesn't support it.
        """
        # Import here to avoid circular imports
        from representation_learning.training.train import Trainer

        # Determine training mode
        training_mode = TrainerFactory._get_training_mode(config)

        # Validate clustering config if present
        if (
            hasattr(config, "clustering_eval")
            and config.clustering_eval
            and config.clustering_eval.enabled
        ):
            # Ensure the model supports extract_embeddings
            if not hasattr(model, "extract_embeddings"):
                raise ValueError(
                    "Model must support extract_embeddings method for clustering "
                    "evaluation. "
                    f"Model type: {type(model).__name__}"
                )
            logger.info("Clustering evaluation configuration validated")

        # Create components
        strategy = StrategyFactory.create_strategy(
            mode=training_mode,
            criterion=build_criterion(config.loss_function),
            device=device,
        )

        checkpoint_manager = CheckpointManager(
            model_dir=config.output_dir,
            checkpoint_freq=getattr(config, "checkpoint_freq", 1),
            experiment_logger=exp_logger,
            run_config=config,
        )

        metrics_tracker = MetricsTracker(
            metrics=config.metrics,
            num_classes=num_classes,
            device=device,
            training_mode=training_mode,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            strategy=strategy,
            checkpoint_manager=checkpoint_manager,
            metrics_tracker=metrics_tracker,
            config=config,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=is_distributed,
            exp_logger=exp_logger,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        return trainer

    @staticmethod
    def _get_training_mode(config: RunConfig) -> str:
        """Determine training mode from configuration.

        Parameters
        ----------
        config : RunConfig
            Training configuration

        Returns
        -------
        str
            Training mode ('supervised', 'clip', 'eat_ssl')
        """
        if config.label_type == "text":
            return "clip"
        elif config.label_type == "self_supervised":
            return "eat_ssl"
        else:
            return "supervised"
