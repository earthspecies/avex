"""
Command-line interface for representation learning experiments.

This module provides a unified CLI for both training and evaluation tasks.
"""

import logging
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Representation learning CLI for training and evaluation."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/run_configs/clip_base.yml",
    help="Path to the training config file",
)
@click.option(
    "--patch",
    "-p",
    "patches",
    type=str,
    multiple=True,
    help="Patch config values in format 'key=value'. Can be used multiple times.",
)
def train(config: Path, patches: tuple[str, ...]) -> None:
    """Train an audio representation model."""
    from representation_learning.run_train import main as train_main

    train_main(config_path=config, patches=patches)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation config file",
)
@click.option(
    "--patch",
    "-p",
    "patches",
    type=str,
    multiple=True,
    help="Patch config values in format 'key=value'. Can be used multiple times.",
)
def evaluate(config: Path, patches: tuple[str, ...]) -> None:
    """Run linear-probe / fine-tuning experiments."""
    from representation_learning.run_evaluate import main as eval_main

    eval_main(config_path=config, patches=patches)


if __name__ == "__main__":
    cli()
