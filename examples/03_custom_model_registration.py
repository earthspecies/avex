"""
Example 3: Custom Model Registration and Plugin Architecture

This example demonstrates:
- Creating custom model classes
- Registering them with the plugin architecture
- Using custom models with the API
- Model class management functions
"""

import argparse

import torch
import torch.nn as nn

from representation_learning import (
    build_model,
    create_model,
    get_model_class,
    list_model_classes,
    register_model_class,
)
from representation_learning.models.base_model import ModelBase


# Example 1: Simple CNN Model
@register_model_class
class SimpleAudioCNN(ModelBase):
    """A simple CNN for audio classification."""

    name = "simple_audio_cnn"

    def __init__(self, device: str, num_classes: int, audio_config: dict = None, **kwargs: object) -> None:
        super().__init__(device=device, audio_config=audio_config)

        # Simple CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # Fixed size output
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
        """Forward pass through the model.

        Returns:
            torch.Tensor: Model output.
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Apply convolutions
        features = self.conv_layers(x)

        # Classify
        output = self.classifier(features)
        return output

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The embedding dimension.
        """
        return 256


# Example 2: Transformer-based Model
@register_model_class
class SimpleAudioTransformer(ModelBase):
    """A simple transformer for audio classification."""

    name = "simple_audio_transformer"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: dict = None,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

        self.to(device)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the model.

        Returns:
            torch.Tensor: Model output.
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc

        # Create padding mask if provided
        if padding_mask is not None:
            # Convert to transformer format (True = ignore)
            mask = ~padding_mask.bool()
        else:
            mask = None

        # Apply transformer
        features = self.transformer(x, src_key_padding_mask=mask)

        # Global average pooling
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(features)
            features = features.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = features.sum(dim=1) / lengths
        else:
            pooled = features.mean(dim=1)

        # Classify
        output = self.classifier(pooled)
        return output

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The embedding dimension.
        """
        return self.d_model


# Example 3: MLP Model with custom parameters
@register_model_class
class SimpleAudioMLP(ModelBase):
    """A simple MLP for audio classification."""

    name = "simple_audio_mlp"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: dict = None,
        hidden_dims: list = None,
        dropout: float = 0.2,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Build MLP layers
        layers = []
        input_dim = 16000  # Assume fixed input size for simplicity

        for _i, hidden_dim in enumerate(hidden_dims):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim

        # Final classifier
        layers.append(nn.Linear(input_dim, num_classes))

        self.mlp = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the model.

        Returns:
            torch.Tensor: Model output.
        """
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Ensure correct input size (pad or truncate)
        if x.size(1) != 16000:
            if x.size(1) < 16000:
                # Pad
                padding = torch.zeros(x.size(0), 16000 - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                # Truncate
                x = x[:, :16000]

        return self.mlp(x)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The embedding dimension.
        """
        return 128  # Last hidden layer size


def main(device: str = "cpu") -> None:
    print("🚀 Example 3: Custom Model Registration and Plugin Architecture")
    print("=" * 70)

    # Example 1: List registered model classes
    print("\n📋 Registered Model Classes:")
    model_classes = list_model_classes()
    for cls_name in model_classes:
        print(f"  - {cls_name}")

    # Example 2: Test custom CNN model
    print("\n🔧 Testing Simple Audio CNN:")
    try:
        model = create_model("simple_audio_cnn", num_classes=10, device=device)
        print(f"✅ Created CNN model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Embedding dim: {model.get_embedding_dim()}")

        # Test forward pass
        dummy_input = torch.randn(2, 16000, device=device)  # 2 samples, 1 second each
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    except Exception as e:
        print(f"❌ Error with CNN model: {e}")

    # Example 3: Test custom Transformer model
    print("\n🔧 Testing Simple Audio Transformer:")
    try:
        model = create_model(
            "simple_audio_transformer",
            num_classes=15,
            device=device,
            d_model=64,  # Custom parameter
            nhead=4,  # Custom parameter
            num_layers=2,  # Custom parameter
        )
        print(f"✅ Created Transformer model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Embedding dim: {model.get_embedding_dim()}")

        # Test forward pass
        dummy_input = torch.randn(2, 1000, device=device)  # 2 samples, 1000 timesteps
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    except Exception as e:
        print(f"❌ Error with Transformer model: {e}")

    # Example 4: Test custom MLP model
    print("\n🔧 Testing Simple Audio MLP:")
    try:
        model = create_model(
            "simple_audio_mlp",
            num_classes=20,
            device=device,
            hidden_dims=[256, 128, 64],  # Custom architecture
            dropout=0.3,  # Custom dropout
        )
        print(f"✅ Created MLP model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Embedding dim: {model.get_embedding_dim()}")

        # Test forward pass
        dummy_input = torch.randn(3, 8000, device=device)  # 3 samples, shorter audio
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    except Exception as e:
        print(f"❌ Error with MLP model: {e}")

    # Example 5: Model class management
    print("\n📊 Model Class Management:")
    try:
        # Check if a model class is registered
        model_class = get_model_class("simple_audio_cnn")
        is_registered = model_class is not None
        print(f"simple_audio_cnn registered: {is_registered}")

        # Get a specific model class
        if model_class is not None:
            print(f"Model class: {model_class.__name__}")

        # Create model using build_model (alternative to create_model)
        model = build_model("simple_audio_cnn", device=device, num_classes=5)
        print(f"✅ Built model with build_model: {type(model).__name__}")

    except Exception as e:
        print(f"❌ Error in model class management: {e}")

    # Example 6: Unregister a model class
    print("\n🗑️ Unregistering Model Class:")
    try:
        # Note: Model classes remain registered for the session
        # To use different configurations, register with unique names
        print("✅ Model classes remain registered for the session")
        print("   To use different configurations, register with unique names")

        # Check if it's still registered
        model_class = get_model_class("simple_audio_mlp")
        is_registered = model_class is not None
        print(f"simple_audio_mlp still registered: {is_registered}")

        # Try to create it (should fail)
        try:
            model = create_model("simple_audio_mlp", num_classes=10, device=device)
            print("❌ Unexpected: Model creation succeeded after unregistering")
        except Exception as e:
            print(f"✅ Expected error after unregistering: {e}")

    except Exception as e:
        print(f"❌ Error unregistering model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Model Registration Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for model and data (default: cpu)",
    )
    args = parser.parse_args()
    main(device=args.device)
