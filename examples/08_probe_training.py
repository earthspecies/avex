"""
Example 8: Probe Training

This example demonstrates the probe API:
- Listing available probes
- Building probes with base models
- Creating probes for training
- Using different probe configurations

Audio Requirements:
- Each model expects a specific sample rate (defined in model_spec.audio_config.sample_rate)
- Check with: describe_model("model_name") or get_model_spec("model_name").audio_config.sample_rate
"""

import argparse

import torch

from representation_learning.api import build_probe_from_config, load_model
from representation_learning.configs import ProbeConfig


def main(device: str = "cpu") -> None:
    """Demonstrate probe API functionality."""
    print("Example 8: Probe Training")
    print("=" * 60)

    # =========================================================================
    # Part 1: Build probe with base model (online mode)
    # =========================================================================
    print("\nPart 1: Build Probe with Base Model (Online Mode)")
    print("-" * 60)

    # Load base model
    print("Loading base model: beats_naturelm")
    base_model = load_model("beats_naturelm", return_features_only=True, device=device)
    print(f"  Base model loaded: {type(base_model).__name__}")

    # Define a simple linear probe config
    print("\nBuilding linear probe (backbone features)...")
    probe_config = ProbeConfig(
        probe_type="linear",
        target_layers=["backbone"],
        aggregation="mean",
        freeze_backbone=True,
        online_training=True,
    )
    probe = build_probe_from_config(
        probe_config=probe_config,
        base_model=base_model,
        num_classes=10,
        device=device,
    )
    print(f"  Probe loaded: {type(probe).__name__}")
    print(f"  Total parameters: {sum(p.numel() for p in probe.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 16000 * 3, device=device)
    with torch.no_grad():
        output = probe(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # =========================================================================
    # Part 4: Build probe for offline training
    # =========================================================================
    print("\nPart 4: Build Probe for Offline Training")
    print("-" * 60)

    # Simulate pre-computed embeddings
    embedding_dim = 768  # BEATs embedding dimension
    print(f"Simulating pre-computed embeddings with dim={embedding_dim}")

    # Build probe without base model using a config
    offline_config = ProbeConfig(
        probe_type="linear",
        target_layers=["backbone"],
        aggregation="none",
        freeze_backbone=True,
        online_training=False,
    )
    probe_offline = build_probe_from_config(
        probe_config=offline_config,
        base_model=None,
        num_classes=10,
        device=device,
        feature_mode=True,
        input_dim=embedding_dim,
    )
    print(f"  Offline probe loaded: {type(probe_offline).__name__}")
    print(f"  Total parameters: {sum(p.numel() for p in probe_offline.parameters()):,}")

    # Test with embeddings
    dummy_embeddings = torch.randn(2, embedding_dim, device=device)
    with torch.no_grad():
        output = probe_offline(dummy_embeddings)
    print(f"  Embedding shape: {dummy_embeddings.shape}")
    print(f"  Output shape: {output.shape}")

    # =========================================================================
    # Part 5: Build different probe types
    # =========================================================================
    print("\nPart 5: Build Different Probe Types")
    print("-" * 60)

    probe_types = [
        ("linear", {"aggregation": "mean"}),
        ("mlp", {"aggregation": "mean", "hidden_dims": [512, 256]}),
        ("attention", {"input_processing": "sequence", "num_heads": 4, "attention_dim": 128}),
    ]

    for probe_type, extra_cfg in probe_types:
        cfg = ProbeConfig(
            probe_type=probe_type,
            target_layers=["backbone"],
            freeze_backbone=True,
            online_training=True,
            **extra_cfg,
        )
        probe = build_probe_from_config(
            probe_config=cfg,
            base_model=base_model,
            num_classes=10,
            device=device,
        )
        param_count = sum(p.numel() for p in probe.parameters())
        print(f"  {probe_type:<20}: {param_count:>10,} parameters")

    # =========================================================================
    # Part 6: Build probe with loaded base model
    # =========================================================================
    print("\nPart 6: Build MLP Probe with Loaded Base Model")
    print("-" * 60)

    # Build probe with the already-loaded base model using a custom config
    mlp_config = ProbeConfig(
        probe_type="mlp",
        target_layers=["backbone"],
        aggregation="mean",
        hidden_dims=[1024, 512],
        freeze_backbone=True,
        online_training=True,
    )
    probe = build_probe_from_config(
        probe_config=mlp_config,
        base_model=base_model,
        num_classes=25,
        device=device,
    )
    print(f"  Probe built: {type(probe).__name__}")
    print(f"  Total parameters: {sum(p.numel() for p in probe.parameters()):,}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Key Functions")
    print("=" * 60)
    print("""
- list_probes(): List all available probes with details
- describe_probe(name): Get detailed probe information
- build_probe(name, ...): Build probe with base model or for offline mode

Probe Naming Convention:
  {probe_type}_last  - Uses final backbone layer (fast, simple)
  {probe_type}_all   - Uses all layers (multi-scale, expressive)

Online mode (with base model):
  base = load_model("beats_naturelm", return_features_only=True)
  probe = build_probe("linear_last", base_model=base, num_classes=50, device="cpu")

Offline mode (pre-computed embeddings):
  probe = build_probe("linear_last", base_model=None, num_classes=50,
                      device="cpu", feature_mode=True, input_dim=768)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Training Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (default: cpu)",
    )
    args = parser.parse_args()
    main(device=args.device)
