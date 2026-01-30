"""
Example 7: Probe Training and Inference

This example demonstrates a complete workflow:
- Loading a pretrained backbone model
- Attaching a linear probe head
- Training the probe for a few epochs
- Saving the trained model checkpoint
- Loading the checkpoint for inference

Audio Requirements:
- Each model expects a specific sample rate (defined in model_spec.audio_config.sample_rate)
- Check with: describe_model("model_name") or get_model_spec("model_name").audio_config.sample_rate
- For full reproducibility, resample using librosa with these exact parameters:

    import librosa
    audio_resampled = librosa.resample(
        audio, orig_sr=original_sr, target_sr=target_sr,
        res_type="kaiser_best", scale=True
    )
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from avex import load_model
from avex.configs import ProbeConfig
from avex.models.probes.utils import build_probe_from_config


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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (audio, labels) in enumerate(dataloader):
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(audio, padding_mask=None)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 5 == 0:  # Print more frequently for smaller dataset
            print(f"   Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the model.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(device)
            labels = labels.to(device)

            outputs = model(audio, padding_mask=None)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main(device: str = "cpu") -> None:
    """Demonstrate probe training and inference workflow."""
    print("Example 7: Probe Training and Inference")
    print("=" * 60)

    # Ensure checkpoints directory exists
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Part 1: Load pretrained backbone and attach probe
    # =========================================================================
    print("\nPart 1: Load Backbone and Attach Probe")
    print("-" * 60)

    # Load a pretrained backbone model (without classifier)
    print("Loading esp_aves2_naturelm_audio_v1_beats backbone...")
    print("   (This may take ~20-30 seconds to download pretrained weights on first run)")
    start_time = time.time()
    # esp_aves2_naturelm_audio_v1_beats has no classifier checkpoint, so it automatically loads in embedding mode
    backbone = load_model("esp_aves2_naturelm_audio_v1_beats", device=device)
    backbone.eval()
    load_time = time.time() - start_time
    print(f"   ✓ Loaded in {load_time:.2f} seconds")
    print(f"   Backbone type: {type(backbone).__name__}")
    print(f"   Backbone parameters: {sum(p.numel() for p in backbone.parameters()):,}")

    # Attach an attention probe head on all layers
    num_classes = 10
    # Note: When using "all" layers, different layers may have different dimensions.
    # The probe will automatically project them to a common dimension.
    # Using num_heads=1 ensures compatibility with any dimension.
    probe_config = ProbeConfig(
        probe_type="attention",
        target_layers=["all"],
        aggregation="mean",
        input_processing="sequence",  # Attention probe works with sequence inputs
        freeze_backbone=True,
        online_training=True,
        num_heads=1,  # Safe choice that works with any feature dimension
        attention_dim=768,  # Target dimension (probe will project if needed)
        num_layers=1,  # Number of attention layers (reduced for speed)
        dropout_rate=0.1,
    )

    print(f"\nAttaching attention probe for {num_classes} classes...")
    print("   Target layers: all")
    print(f"   Attention heads: {probe_config.num_heads}")
    print(f"   Attention dimension: {probe_config.attention_dim}")
    print(f"   Number of layers: {probe_config.num_layers}")
    start_time = time.time()
    model = build_probe_from_config(
        probe_config=probe_config,
        base_model=backbone,
        num_classes=num_classes,
        device=device,
    )
    probe_time = time.time() - start_time
    print(f"   ✓ Probe attached in {probe_time:.2f} seconds")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # =========================================================================
    # Part 2: Prepare training data
    # =========================================================================
    print("\nPart 2: Prepare Training Data")
    print("-" * 60)

    start_time = time.time()
    # Get audio config from backbone to create properly sized dummy data
    from avex import get_model_spec

    model_spec = get_model_spec("esp_aves2_naturelm_audio_v1_beats")
    sample_rate = model_spec.audio_config.sample_rate
    target_length_seconds = model_spec.audio_config.target_length_seconds

    # Create dummy datasets (reduced sizes for memory efficiency with "all" layers)
    # Using "all" layers extracts embeddings from many layers, which is memory-intensive
    train_audio, train_labels = create_dummy_dataset(
        num_samples=20,  # Reduced for memory efficiency with "all" layers
        num_classes=num_classes,
        sample_rate=sample_rate,
        duration_seconds=target_length_seconds,
    )
    val_audio, val_labels = create_dummy_dataset(
        num_samples=10,  # Reduced for memory efficiency
        num_classes=num_classes,
        sample_rate=sample_rate,
        duration_seconds=target_length_seconds,
    )
    test_audio, test_labels = create_dummy_dataset(
        num_samples=10,  # Reduced for memory efficiency
        num_classes=num_classes,
        sample_rate=sample_rate,
        duration_seconds=target_length_seconds,
    )

    train_dataset = TensorDataset(train_audio, train_labels)
    val_dataset = TensorDataset(val_audio, val_labels)
    test_dataset = TensorDataset(test_audio, test_labels)

    # Use smaller batch size with "all" layers to reduce memory usage
    batch_size = 2  # Small batch size for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_time = time.time() - start_time
    print(f"   ✓ Data prepared in {data_time:.2f} seconds")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Target length: {target_length_seconds} seconds")

    # =========================================================================
    # Part 3: Train the probe
    # =========================================================================
    print("\nPart 3: Train Probe")
    print("-" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2  # Reduced from 3 for faster execution
    print(f"Training for {num_epochs} epochs...")

    train_start_time = time.time()
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   Epoch time: {epoch_time:.2f} seconds")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    total_train_time = time.time() - train_start_time
    print(f"\n✓ Training completed in {total_train_time:.2f} seconds")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")

    # =========================================================================
    # Part 4: Save checkpoint
    # =========================================================================
    print("\nPart 4: Save Checkpoint")
    print("-" * 60)

    start_time = time.time()
    checkpoint_path = checkpoints_dir / "trained_probe_checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "best_val_acc": best_val_acc,
        "num_classes": num_classes,
        "probe_config": probe_config.model_dump() if hasattr(probe_config, "model_dump") else None,
    }
    torch.save(checkpoint, checkpoint_path)
    save_time = time.time() - start_time
    print(f"   ✓ Saved checkpoint in {save_time:.2f} seconds")
    print(f"   Checkpoint path: {checkpoint_path}")

    # =========================================================================
    # Part 5: Load checkpoint and perform inference
    # =========================================================================
    print("\nPart 5: Load Checkpoint and Perform Inference")
    print("-" * 60)

    # Load the backbone again (same as before)
    # In practice, you might load it fresh, but for this example we'll reuse it
    print("Using backbone (reusing from training)...")
    start_time = time.time()
    # esp_aves2_naturelm_audio_v1_beats has no classifier checkpoint, so it automatically loads in embedding mode
    loaded_backbone = backbone  # Reuse the same backbone instance
    loaded_backbone.eval()
    backbone_time = time.time() - start_time
    print(f"   ✓ Backbone ready in {backbone_time:.2f} seconds")

    # Rebuild the probe with the same configuration
    print("Rebuilding probe...")
    start_time = time.time()
    loaded_model = build_probe_from_config(
        probe_config=probe_config,
        base_model=loaded_backbone,
        num_classes=num_classes,
        device=device,
    )
    probe_rebuild_time = time.time() - start_time
    print(f"   ✓ Probe rebuilt in {probe_rebuild_time:.2f} seconds")

    # Load the checkpoint weights
    print("Loading checkpoint weights...")
    start_time = time.time()
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    loaded_model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
    loaded_model.eval()
    load_time = time.time() - start_time
    print(f"   ✓ Checkpoint loaded in {load_time:.2f} seconds")
    print(f"   Loaded checkpoint from epoch {checkpoint_data['epoch']}")
    print(f"   Best validation accuracy: {checkpoint_data['best_val_acc']:.2f}%")

    # Perform inference on test set
    print("\nPerforming inference on test set...")
    start_time = time.time()
    test_loss, test_acc = evaluate(loaded_model, test_loader, criterion, device)
    inference_time = time.time() - start_time
    print(f"   ✓ Inference completed in {inference_time:.2f} seconds")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")

    # =========================================================================
    # Part 6: Alternative: Load using load_model with checkpoint_path
    # =========================================================================
    print("\nPart 6: Alternative Loading Method")
    print("-" * 60)

    # Note: This approach requires the model to be registered or using a ModelSpec
    # For this example, we'll show the manual loading approach above
    print("Alternative: You can also use load_model() with checkpoint_path")
    print("   However, this requires the model to be registered in the registry")
    print("   or using a ModelSpec. For probe models, manual loading is recommended.")

    # Clean up
    print("\nCleaning up checkpoint file...")
    checkpoint_path.unlink()
    print("Done")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. Load pretrained backbone:
   - Use load_model() for backbone-only models (automatically loads in embedding mode)

2. Attach probe head:
   - Use build_probe_from_config() to attach a task-specific probe
   - Probe can be linear, MLP, LSTM, attention, or transformer
   - Attention probes work well with "all" layers for richer representations

3. Train the probe:
   - Only probe parameters are trainable (backbone is frozen)
   - Standard PyTorch training loop works

4. Save checkpoint:
   - Save model.state_dict() along with training metadata

5. Load for inference:
   - Rebuild the same model structure
   - Load checkpoint weights with load_state_dict()
   - Use strict=False to handle partial matches
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Training and Inference Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
