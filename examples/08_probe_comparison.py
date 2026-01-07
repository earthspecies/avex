"""
Example 8: Probe Comparison and Offline Training

This example demonstrates:
- Comparing different probe types (linear, MLP, attention, LSTM)
- Comparing different configurations (target_layers, aggregation methods)
- Offline training workflow with pre-computed embeddings
- When to use online vs offline training

This complements Example 7, which shows full online training workflow.

Audio Requirements:
- Each model expects a specific sample rate (defined in model_spec.audio_config.sample_rate)
- Check with: get_model_spec("model_name").audio_config.sample_rate
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from representation_learning import get_model_spec, load_model
from representation_learning.configs import ProbeConfig
from representation_learning.models.probes.utils import (
    build_probe_from_config,
)


def create_dummy_dataset(
    num_samples: int,
    num_classes: int,
    sample_rate: int = 16000,
    duration_seconds: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy dataset for demonstration.

    Args:
        num_samples: Number of samples in the dataset
        num_classes: Number of classes
        sample_rate: Audio sample rate
        duration_seconds: Duration of each audio sample in seconds

    Returns:
        Tuple of (audio_data, labels)
    """
    audio_length = int(sample_rate * duration_seconds)
    audio_data = torch.randn(num_samples, audio_length)
    labels = torch.randint(0, num_classes, (num_samples,))
    return audio_data, labels


def compute_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    target_layers: list[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute embeddings from a backbone model.

    Args:
        model: Backbone model for feature extraction
        dataloader: DataLoader with audio data
        device: Device to use
        target_layers: List of target layers for embedding extraction

    Returns:
        Tuple of (embeddings, labels)
    """
    model.eval()

    # Register hooks if needed
    if hasattr(model, "register_hooks_for_layers"):
        if target_layers is None:
            target_layers = ["last_layer"]
        model.register_hooks_for_layers(target_layers)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(device)
            # Extract embeddings
            if hasattr(model, "extract_embeddings"):
                emb = model.extract_embeddings(audio, aggregation="mean", freeze_backbone=True)
            else:
                emb = model(audio)
            # Handle list of embeddings (from multiple layers)
            if isinstance(emb, list):
                # Stack and mean pool if multiple layers
                if len(emb) > 1:
                    # All should have same batch size, stack and mean
                    stacked = torch.stack([e.mean(dim=1) if e.dim() == 3 else e for e in emb], dim=0)
                    emb = stacked.mean(dim=0)
                else:
                    emb = emb[0]
            # Flatten if needed (handle 3D tensors: batch, seq, features)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)  # Mean pool over sequence dimension
            elif emb.dim() > 3:
                emb = emb.view(emb.shape[0], -1)
            embeddings_list.append(emb.cpu())
            labels_list.append(labels)

    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)


def train_probe_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    feature_mode: bool = False,
) -> tuple[float, float]:
    """Train probe for one epoch.

    Args:
        model: Probe model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        feature_mode: Whether inputs are pre-computed embeddings

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for audio_or_emb, labels in dataloader:
        audio_or_emb = audio_or_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if feature_mode:
            # Input is pre-computed embeddings
            outputs = model(audio_or_emb)
        else:
            # Input is raw audio
            outputs = model(audio_or_emb, padding_mask=None)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main(device: str = "cpu") -> None:
    """Demonstrate probe comparison and offline training."""
    print("Example 8: Probe Comparison and Offline Training")
    print("=" * 60)

    # =========================================================================
    # Part 1: Compare different probe types
    # =========================================================================
    print("\nPart 1: Compare Different Probe Types")
    print("-" * 60)

    # Load backbone once
    print("Loading backbone: beats_naturelm...")
    backbone = load_model("beats_naturelm", device=device)
    backbone.eval()
    print(f"  Backbone loaded: {type(backbone).__name__}")

    # Get model spec for audio config
    model_spec = get_model_spec("beats_naturelm")
    sample_rate = model_spec.audio_config.sample_rate
    target_length_seconds = model_spec.audio_config.target_length_seconds

    # Create dummy data
    num_classes = 10
    dummy_audio, _ = create_dummy_dataset(2, num_classes, sample_rate, target_length_seconds)
    dummy_audio = dummy_audio.to(device)

    # Compare probe types - build each probe separately
    print("\nComparing probe types (same backbone, same target layer):")
    print(f"{'Probe Type':<15} {'Parameters':<15} {'Output Shape':<20}")
    print("-" * 50)

    # Linear probe
    try:
        linear_config = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )
        linear_probe = build_probe_from_config(
            probe_config=linear_config,
            base_model=backbone,
            num_classes=num_classes,
            device=device,
        )
        param_count = sum(p.numel() for p in linear_probe.parameters())
        linear_probe.eval()
        with torch.no_grad():
            output = linear_probe(dummy_audio, padding_mask=None)
        print(f"{'Linear':<15} {param_count:>14,} {str(output.shape):<20}")
    except Exception as e:
        print(f"{'Linear':<15} {'ERROR':<15} {str(e)[:40]:<20}")

    # MLP probe - reload backbone to avoid hook conflicts
    try:
        mlp_backbone = load_model("beats_naturelm", device=device)
        mlp_backbone.eval()
        mlp_config = ProbeConfig(
            probe_type="mlp",
            target_layers=["last_layer"],
            aggregation="mean",
            hidden_dims=[512, 256],
            freeze_backbone=True,
            online_training=True,
        )
        mlp_probe = build_probe_from_config(
            probe_config=mlp_config,
            base_model=mlp_backbone,
            num_classes=num_classes,
            device=device,
        )
        param_count = sum(p.numel() for p in mlp_probe.parameters())
        mlp_probe.eval()
        with torch.no_grad():
            output = mlp_probe(dummy_audio, padding_mask=None)
        print(f"{'MLP':<15} {param_count:>14,} {str(output.shape):<20}")
    except Exception as e:
        print(f"{'MLP':<15} {'ERROR':<15} {str(e)[:40]:<20}")

    # Note: Attention and LSTM probes require sequence processing
    # and may need specific configurations. See Example 7 for attention probe usage.
    print(f"{'Attention':<15} {'See Ex. 7':<15} {'(sequence)':<20}")
    print(f"{'LSTM':<15} {'See docs':<15} {'(sequence)':<20}")

    # =========================================================================
    # Part 2: Compare target layer configurations
    # =========================================================================
    print("\nPart 2: Compare Target Layer Configurations")
    print("-" * 60)

    print("\nComparing target layer configurations (linear probe):")
    print(f"{'Target Layers':<20} {'Parameters':<15} {'Output Shape':<20}")
    print("-" * 55)

    # last_layer configuration
    try:
        last_layer_config = ProbeConfig(
            probe_type="linear",
            target_layers=["last_layer"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )
        last_layer_probe = build_probe_from_config(
            probe_config=last_layer_config,
            base_model=backbone,
            num_classes=num_classes,
            device=device,
        )
        param_count = sum(p.numel() for p in last_layer_probe.parameters())
        last_layer_probe.eval()
        with torch.no_grad():
            output = last_layer_probe(dummy_audio, padding_mask=None)
        print(f"{'last_layer':<20} {param_count:>14,} {str(output.shape):<20}")
    except Exception as e:
        print(f"{'last_layer':<20} {'ERROR':<15} {str(e)[:40]:<20}")

    # all layers configuration - reload backbone to avoid hook conflicts
    try:
        all_layers_backbone = load_model("beats_naturelm", device=device)
        all_layers_backbone.eval()
        all_layers_config = ProbeConfig(
            probe_type="linear",
            target_layers=["all"],
            aggregation="mean",
            freeze_backbone=True,
            online_training=True,
        )
        all_layers_probe = build_probe_from_config(
            probe_config=all_layers_config,
            base_model=all_layers_backbone,
            num_classes=num_classes,
            device=device,
        )
        param_count = sum(p.numel() for p in all_layers_probe.parameters())
        all_layers_probe.eval()
        with torch.no_grad():
            output = all_layers_probe(dummy_audio, padding_mask=None)
        print(f"{'all layers':<20} {param_count:>14,} {str(output.shape):<20}")
    except Exception as e:
        print(f"{'all layers':<20} {'ERROR':<15} {str(e)[:40]:<20}")
        print("  Note: 'all' layers extracts from many layers and may be memory-intensive")

    # =========================================================================
    # Part 3: Offline training workflow
    # =========================================================================
    print("\nPart 3: Offline Training Workflow")
    print("-" * 60)
    print("""
Offline training is useful when:
- You want to train multiple probes on the same embeddings
- You need to save memory by pre-computing embeddings once
- You want to experiment with different probe architectures quickly
- Your dataset is large and embedding computation is expensive
""")

    # Step 1: Pre-compute embeddings
    print("\nStep 1: Pre-compute embeddings from backbone...")
    train_audio, train_labels = create_dummy_dataset(20, num_classes, sample_rate, target_length_seconds)
    val_audio, val_labels = create_dummy_dataset(10, num_classes, sample_rate, target_length_seconds)

    train_dataset = TensorDataset(train_audio, train_labels)
    val_dataset = TensorDataset(val_audio, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    start_time = time.time()
    train_embeddings, train_labels_emb = compute_embeddings(
        backbone, train_loader, device, target_layers=["last_layer"]
    )
    val_embeddings, val_labels_emb = compute_embeddings(backbone, val_loader, device, target_layers=["last_layer"])
    embedding_time = time.time() - start_time

    embedding_dim = train_embeddings.shape[1]
    print(f"  ✓ Embeddings computed in {embedding_time:.2f} seconds")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Val embeddings shape: {val_embeddings.shape}")

    # Step 2: Build probe in offline mode
    print("\nStep 2: Build probe in offline mode (no base model)...")
    offline_probe_config = ProbeConfig(
        probe_type="mlp",
        target_layers=["backbone"],  # Not used in offline mode, but required
        aggregation="none",  # Not used in offline mode
        freeze_backbone=True,
        online_training=False,
        hidden_dims=[256, 128],
    )

    offline_probe = build_probe_from_config(
        probe_config=offline_probe_config,
        input_dim=embedding_dim,
        num_classes=num_classes,
        device=device,
    )
    print(f"  ✓ Offline probe built: {type(offline_probe).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in offline_probe.parameters()):,}")

    # Step 3: Train probe on pre-computed embeddings
    print("\nStep 3: Train probe on pre-computed embeddings...")
    train_emb_dataset = TensorDataset(train_embeddings, train_labels_emb)
    val_emb_dataset = TensorDataset(val_embeddings, val_labels_emb)
    train_emb_loader = DataLoader(train_emb_dataset, batch_size=4, shuffle=True)
    val_emb_loader = DataLoader(val_emb_dataset, batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(offline_probe.parameters(), lr=0.001)

    num_epochs = 2
    print(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_probe_epoch(
            offline_probe, train_emb_loader, optimizer, criterion, device, feature_mode=True
        )
        offline_probe.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for emb, labels in val_emb_loader:
                emb = emb.to(device)
                labels = labels.to(device)
                outputs = offline_probe(emb)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss_total / len(val_emb_loader)
        val_acc = 100 * val_correct / val_total

        print(f"  Epoch {epoch + 1}/{num_epochs}:")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # =========================================================================
    # Part 4: Compare online vs offline training
    # =========================================================================
    print("\nPart 4: Online vs Offline Training Comparison")
    print("-" * 60)

    print("""
Online Training (Example 7):
  - Backbone and probe are connected
  - Embeddings computed on-the-fly during training
  - More memory usage (backbone + probe in memory)
  - Slower per epoch (embedding computation each time)
  - Better for: Single probe training, when you need gradients through backbone

Offline Training (This example):
  - Embeddings pre-computed once
  - Probe trained on saved embeddings
  - Less memory (only probe in memory during training)
  - Faster per epoch (no embedding computation)
  - Better for: Multiple probe experiments, large datasets, memory constraints
""")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. Probe Types:
   - Linear: Fastest, simplest, good baseline
   - MLP: More expressive, can learn non-linear patterns
   - Attention: Good for sequence data, can attend to important features
   - LSTM: Good for temporal patterns, sequential processing

2. Target Layers:
   - "last_layer": Fast, uses final layer features
   - "all": Slower, uses multi-scale features from all layers

3. Training Modes:
   - Online: Backbone + probe together, embeddings computed on-the-fly
   - Offline: Embeddings pre-computed, probe trained separately

4. When to use offline mode:
   - Training multiple probes on same embeddings
   - Memory constraints
   - Large datasets where embedding computation is expensive
   - Quick experimentation with probe architectures
""")

    print("\n✓ Example completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Comparison and Offline Training Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
