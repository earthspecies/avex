#!/usr/bin/env python3
"""
CLI helper to list available models in the registry.

Usage:
    python -m avex.api.list_models
    python -m avex.api.list_models --detailed
"""

import sys

import click

from avex.models.utils.registry import list_models


def print_models(detailed: bool = False) -> None:
    """Print available models in the registry.

    Args:
        detailed: If True, show detailed model information
    """
    models = list_models()

    if not models:
        print("No models registered in the registry.")
        return

    print(f"Available models ({len(models)}):")
    print("=" * 50)

    for name, model_spec in models.items():
        print(f"\n📦 {name}")

        if detailed:
            print(f"  Name: {model_spec.name}")
            print(f"  Pretrained: {model_spec.pretrained}")
            print(f"  Device: {model_spec.device}")

            if hasattr(model_spec, "audio_config") and model_spec.audio_config:
                audio_cfg = model_spec.audio_config
                print("  Audio Config:")
                print(f"    Sample Rate: {audio_cfg.sample_rate}")
                print(f"    Representation: {audio_cfg.representation}")
                print(f"    Target Length: {audio_cfg.target_length_seconds}s")
                print(f"    Normalize: {audio_cfg.normalize}")

            if hasattr(model_spec, "use_naturelm"):
                print(f"  NatureLM: {model_spec.use_naturelm}")
        else:
            # Simple format
            pretrained_str = "✓" if model_spec.pretrained else "✗"
            print(f"  Pretrained: {pretrained_str} | Device: {model_spec.device}")


@click.command()
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed model information",
)
def main(detailed: bool) -> None:
    """List available models in the representation learning registry."""
    try:
        print_models(detailed=detailed)
    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
