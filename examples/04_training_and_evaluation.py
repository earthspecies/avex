"""
Example 4: Training and Evaluation Workflows

This example demonstrates:
- Complete training workflow with custom models
- Model evaluation and testing
- Working with different model types
- Best practices for model development

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
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from avex import load_model, register_model_class
from avex.configs import ProbeConfig
from avex.models.base_model import ModelBase
from avex.models.probes.utils import build_probe_from_config

# =============================================================================
# Custom Training Model
# =============================================================================


@register_model_class
class TrainingExampleModel(ModelBase):
    """A simple model for demonstrating training workflows."""

    name = "training_example"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: Optional[dict] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the model."""
        super().__init__(device=device, audio_config=audio_config)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self.to(device)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns
        -------
        int
            Embedding dimension.
        """
        return 256


# =============================================================================
# Helper Functions
# =============================================================================


def create_dummy_dataset(
    num_samples: int = 1000, num_classes: int = 10, audio_length: int = 16000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy dataset for training.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (audio_data, labels).
    """
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

    Returns
    -------
    tuple[float, float]
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.0 * correct / total:.2f}%")

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the model.

    Returns
    -------
    tuple[float, float]
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(dataloader), 100.0 * correct / total


# =============================================================================
# Main Example
# =============================================================================


def main(device: str = "cpu") -> None:
    """Demonstrate training and evaluation workflows.

    Parameters
    ----------
    device:
        Device identifier to use for all models and tensors (for example
        ``\"cpu\"`` or ``\"cuda:0\"``).

    Raises
    ------
    ValueError
        If an unknown model name is encountered in the comparison section.
    """
    print("Example 4: Training and Evaluation Workflows")
    print("=" * 60)
    print(f"Using device: {device}")

    # =========================================================================
    # Part 1: Training from scratch
    # =========================================================================
    print("\nPart 1: Training Custom Model from Scratch")
    print("-" * 60)

    model = TrainingExampleModel(device=device, num_classes=10)
    print(f"Created model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_data, train_labels = create_dummy_dataset(800, 10)
    val_data, val_labels = create_dummy_dataset(200, 10)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print("\nTraining completed.")

    # =========================================================================
    # Part 2: Model evaluation
    # =========================================================================
    print("\nPart 2: Model Evaluation")
    print("-" * 60)

    model = TrainingExampleModel(device=device, num_classes=8)
    model.eval()

    test_data, test_labels = create_dummy_dataset(100, 8)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Inference speed test
    test_input = torch.randn(1, 16000).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)

    # Time inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    print(f"Average inference time: {avg_inference_time * 1000:.2f}ms")

    # =========================================================================
    # Part 3: Model checkpointing
    # =========================================================================
    print("\nPart 3: Model Checkpointing")
    print("-" * 60)

    model = TrainingExampleModel(device=device, num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Quick training
    train_data, train_labels = create_dummy_dataset(100, 6)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Trained model - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    # Save checkpoint
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoints_dir / "example_checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 1,
        "loss": train_loss,
        "accuracy": train_acc,
        "num_classes": 6,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Load model from checkpoint
    # Create a ModelSpec for the custom model
    # num_classes will be automatically extracted from checkpoint by load_model
    from avex.configs import AudioConfig, ModelSpec

    model_spec = ModelSpec(
        name="training_example",
        pretrained=False,
        device=device,
        audio_config=AudioConfig(sample_rate=16000, representation="raw", target_length_seconds=1.0),
    )
    loaded_model = load_model(model_spec, checkpoint_path=str(checkpoint_path), device=device)
    print(f"Loaded model: {type(loaded_model).__name__}")

    # Verify loaded model
    test_input = torch.randn(1, 16000).to(device)
    with torch.no_grad():
        output = loaded_model(test_input)
    print(f"   Output shape: {output.shape}")

    # Clean up
    checkpoint_path.unlink()
    print("Cleaned up checkpoint file.")

    # =========================================================================
    # Part 4: Linear probe on embeddings (offline mode)
    # =========================================================================
    print("\nPart 4: Linear Probe on Embeddings (Offline Mode)")
    print("-" * 60)

    embedding_dim = 768
    probe_config = ProbeConfig(
        probe_type="linear",
        target_layers=["backbone"],
        aggregation="none",
        freeze_backbone=True,
        online_training=False,
    )

    probe = build_probe_from_config(
        probe_config=probe_config,
        input_dim=embedding_dim,
        num_classes=6,
        device=device,
    )

    # Dummy embedding dataset (simulating pre-computed embeddings)
    embed_data = torch.randn(200, embedding_dim)
    embed_labels = torch.randint(0, 6, (200,))
    embed_dataset = TensorDataset(embed_data, embed_labels)
    embed_loader = DataLoader(embed_dataset, batch_size=32, shuffle=True)

    probe_optimizer = optim.Adam(probe.parameters(), lr=0.001)
    probe_criterion = nn.CrossEntropyLoss()

    print("Training linear probe on embeddings...")
    for epoch in range(2):
        probe.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_embeds, batch_labels in embed_loader:
            batch_embeds = batch_embeds.to(device)
            batch_labels = batch_labels.to(device)

            probe_optimizer.zero_grad()
            outputs = probe(batch_embeds)
            loss = probe_criterion(outputs, batch_labels)
            loss.backward()
            probe_optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_loss = total_loss / len(embed_loader)
        acc = 100.0 * correct / total
        print(f"   Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

    # =========================================================================
    # Part 5: Model comparison
    # =========================================================================
    print("\nPart 4: Model Comparison")
    print("-" * 60)

    models_to_compare = [
        ("training_example", {}),
    ]

    test_data, test_labels = create_dummy_dataset(50, 5)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    print("Comparing models:")
    for model_name, kwargs in models_to_compare:
        if model_name == "training_example":
            model = TrainingExampleModel(device=device, num_classes=5, **kwargs)
        else:
            raise ValueError(f"Unknown model name in models_to_compare: {model_name}")
        model.eval()

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        param_count = sum(p.numel() for p in model.parameters())

        print(f"\n  {model_name}:")
        print(f"    Parameters: {param_count:,}")
        print(f"    Test Accuracy: {test_acc:.2f}%")
        print(f"    Test Loss: {test_loss:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
- Use custom model classes (or backbones + probes) for training new models
- Save checkpoints with model state, optimizer state, and metadata
- Use load_model() with checkpoint_path to resume training
- Compare models with consistent evaluation setup
- Measure inference time for deployment decisions
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and Evaluation Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
