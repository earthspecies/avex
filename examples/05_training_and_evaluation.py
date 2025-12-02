"""
Example 5: Training and Evaluation Workflows

This example demonstrates:
- Complete training workflow with custom models
- Model evaluation and testing
- Working with different model types
- Best practices for model development
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from representation_learning import (
    create_model,
    list_models,
    load_model,
    register_model_class,
)
from representation_learning.models.base_model import ModelBase


# Custom model for training example
@register_model_class
class TrainingExampleModel(ModelBase):
    """A simple model for demonstrating training workflows."""

    name = "training_example"

    def __init__(self, device: str, num_classes: int, audio_config: dict = None, **kwargs: object) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Simple architecture
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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The embedding dimension (256).
        """
        return 256


def create_dummy_dataset(num_samples: int = 1000, num_classes: int = 10, audio_length: int = 16000) -> TensorDataset:
    """Create a dummy dataset for training.

    Returns:
        TensorDataset: A dataset containing random audio data and labels.
    """
    # Generate random audio data
    audio_data = torch.randn(num_samples, audio_length)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return audio_data, labels


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        tuple[float, float]: A tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0
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
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.0 * correct / total:.2f}%")

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the model.

    Returns:
        tuple[float, float]: A tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0
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


def main(device: str = "cpu") -> None:
    print("🚀 Example 5: Training and Evaluation Workflows")
    print("=" * 60)

    # Setup
    print(f"Using device: {device}")

    # Example 1: Training a custom model from scratch
    print("\n🏋️ Training Custom Model from Scratch:")
    try:
        # Create model
        model = create_model("training_example", num_classes=10, device=device)
        print(f"✅ Created model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create dummy dataset
        train_data, train_labels = create_dummy_dataset(800, 10)
        val_data, val_labels = create_dummy_dataset(200, 10)

        # Create data loaders
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 3
        print(f"\n   Training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\n   Epoch {epoch + 1}/{num_epochs}:")
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Error in training: {e}")
        import traceback

        traceback.print_exc()

    # Example 2: Fine-tuning a pre-trained model
    print("\n🔧 Fine-tuning Pre-trained Model:")
    try:
        # Load a pre-trained model (if available)
        print("\n📋 Available models:")
        models = list_models()
        if models:
            model_name = list(models.keys())[0]
            print(f"\n   Using model: {model_name}")

            # Create model for fine-tuning
            model = create_model(model_name, num_classes=5, device=device)
            print(f"✅ Created model for fine-tuning: {type(model).__name__}")

            # Setup for fine-tuning (lower learning rate)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning

            # Create smaller dataset for fine-tuning
            fine_tune_data, fine_tune_labels = create_dummy_dataset(200, 5)
            fine_tune_dataset = TensorDataset(fine_tune_data, fine_tune_labels)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=16, shuffle=True)

            # Fine-tuning loop (fewer epochs)
            print("   Fine-tuning for 2 epochs...")
            for epoch in range(2):
                print(f"\n   Fine-tuning Epoch {epoch + 1}/2:")
                train_loss, train_acc = train_epoch(model, fine_tune_loader, optimizer, criterion, device)
                print(f"   Fine-tune Loss: {train_loss:.4f}, Fine-tune Acc: {train_acc:.2f}%")

            print("✅ Fine-tuning completed!")
        else:
            print("   No pre-trained models available for fine-tuning")

    except Exception as e:
        print(f"❌ Error in fine-tuning: {e}")

    # Example 3: Model evaluation and testing
    print("\n📊 Model Evaluation and Testing:")
    try:
        # Create a test model
        model = create_model("training_example", num_classes=8, device=device)
        model.eval()

        # Create test dataset
        test_data, test_labels = create_dummy_dataset(100, 8)
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Evaluate model
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print("✅ Model evaluation completed:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2f}%")

        # Test inference speed
        import time

        model.eval()
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
        print(f"   Average inference time: {avg_inference_time * 1000:.2f}ms")

    except Exception as e:
        print(f"❌ Error in evaluation: {e}")

    # Example 4: Model checkpointing
    print("\n💾 Model Checkpointing:")
    try:
        # Create and train a model
        model = create_model("training_example", num_classes=6, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Quick training
        train_data, train_labels = create_dummy_dataset(100, 6)
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Train for 1 epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"   Trained model - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Save checkpoint
        from pathlib import Path

        # Ensure checkpoints directory exists
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
        print(f"✅ Saved checkpoint to: {checkpoint_path}")

        # Load model from checkpoint (pass checkpoint_path directly)
        loaded_model = load_model("training_example", checkpoint_path=str(checkpoint_path), device=device)
        print("✅ Loaded model from checkpoint")
        print(f"✅ Loaded model from checkpoint: {type(loaded_model).__name__}")

        # Verify the loaded model works
        test_input = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            output = loaded_model(test_input)
        print(f"   Loaded model output shape: {output.shape}")

        # Clean up
        import os

        os.remove(checkpoint_path)
        print("✅ Cleaned up checkpoint file")

    except Exception as e:
        print(f"❌ Error in checkpointing: {e}")

    # Example 5: Model comparison
    print("\n🔍 Model Comparison:")
    try:
        # Create different models for comparison
        models_to_compare = [
            ("training_example", 5),
            ("simple_audio_cnn", 5),
            ("simple_audio_mlp", 5),
        ]

        test_data, test_labels = create_dummy_dataset(50, 5)
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        print("   Comparing models:")
        for model_name, num_classes in models_to_compare:
            try:
                model = create_model(model_name, num_classes=num_classes, device=device)
                model.eval()

                # Evaluate
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                param_count = sum(p.numel() for p in model.parameters())

                print(f"   - {model_name}:")
                print(f"     Parameters: {param_count:,}")
                print(f"     Test Accuracy: {test_acc:.2f}%")
                print(f"     Test Loss: {test_loss:.4f}")

            except Exception as e:
                print(f"   - {model_name}: Error - {e}")

    except Exception as e:
        print(f"❌ Error in model comparison: {e}")

    print("\n🎉 Training and evaluation examples completed!")


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
