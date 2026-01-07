"""
Example 3: Custom Model Registration and Plugin Architecture

This example demonstrates:
- Creating custom model classes
- When and why to register custom models
- Using custom models with the API
- Model class management functions

**When Do You Need to Register?**

Registration is ONLY required if you want to use the plugin architecture:
- Using build_model() or build_model_from_spec() with ModelSpecs
- Loading models from YAML configuration files
- Dynamic model selection based on configuration

Registration is NOT required if:
- You're instantiating models directly: MyModel(device="cpu", num_classes=10)
- You're using models standalone or attaching probes directly

See docs/custom_model_registration.md for detailed guidance.

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
from typing import Optional

import torch
import torch.nn as nn

from representation_learning import get_model_class, list_model_classes, register_model_class
from representation_learning.models.base_model import ModelBase

# =============================================================================
# Custom Model Definitions
# =============================================================================


@register_model_class
class SimpleAudioCNN(ModelBase):
    """A simple CNN for audio classification."""

    name = "simple_audio_cnn"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: Optional[dict] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the model."""
        super().__init__(device=device, audio_config=audio_config)

        self.conv_layers = nn.Sequential(
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
        features = self.conv_layers(x)
        return self.classifier(features)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns
        -------
        int
            Embedding dimension.
        """
        return 256


@register_model_class
class SimpleAudioTransformer(ModelBase):
    """A simple transformer for audio classification."""

    name = "simple_audio_transformer"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: Optional[dict] = None,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        **kwargs: object,
    ) -> None:
        """Initialize the model."""
        super().__init__(device=device, audio_config=audio_config)

        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
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
            x = x.unsqueeze(-1)

        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)

        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc

        mask = None
        if padding_mask is not None:
            mask = ~padding_mask.bool()

        features = self.transformer(x, src_key_padding_mask=mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(features)
            features = features.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = features.sum(dim=1) / lengths
        else:
            pooled = features.mean(dim=1)

        return self.classifier(pooled)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns
        -------
        int
            Embedding dimension.
        """
        return self.d_model


@register_model_class
class SimpleAudioMLP(ModelBase):
    """A simple MLP for audio classification."""

    name = "simple_audio_mlp"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: Optional[dict] = None,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.2,
        **kwargs: object,
    ) -> None:
        """Initialize the model."""
        super().__init__(device=device, audio_config=audio_config)

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        input_dim = 16000  # Fixed input size

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Ensure correct input size
        if x.size(1) < 16000:
            padding = torch.zeros(x.size(0), 16000 - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > 16000:
            x = x[:, :16000]

        return self.mlp(x)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension.

        Returns
        -------
        int
            Embedding dimension.
        """
        return 128


# =============================================================================
# Main Example
# =============================================================================


def main(device: str = "cpu") -> None:
    """Demonstrate custom model registration."""
    print("Example 3: Custom Model Registration")
    print("=" * 50)

    # =========================================================================
    # Part 1: List registered model classes
    # =========================================================================
    print("\nPart 1: Registered Model Classes")
    print("-" * 50)

    model_classes = list_model_classes()
    print(f"Total registered classes: {len(model_classes)}")
    for cls_name in model_classes:
        print(f"  - {cls_name}")

    # =========================================================================
    # Part 2: Test Simple Audio CNN
    # =========================================================================
    print("\nPart 2: Simple Audio CNN")
    print("-" * 50)

    model = SimpleAudioCNN(device=device, num_classes=10)
    print(f"Created: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Embedding dim: {model.get_embedding_dim()}")

    dummy_input = torch.randn(2, 16000, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    # =========================================================================
    # Part 3: Test Simple Audio Transformer
    # =========================================================================
    print("\nPart 3: Simple Audio Transformer")
    print("-" * 50)

    model = SimpleAudioTransformer(
        device=device,
        num_classes=15,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    print(f"Created: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   d_model: 64, nhead: 4, num_layers: 2")

    dummy_input = torch.randn(2, 1000, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    # =========================================================================
    # Part 4: Test Simple Audio MLP
    # =========================================================================
    print("\nPart 4: Simple Audio MLP")
    print("-" * 50)

    model = SimpleAudioMLP(
        device=device,
        num_classes=20,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
    )
    print(f"Created: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   Hidden dims: [256, 128, 64], dropout: 0.3")

    dummy_input = torch.randn(3, 8000, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    # =========================================================================
    # Part 5: Model class management
    # =========================================================================
    print("\nPart 5: Model Class Management")
    print("-" * 50)

    # Check registration
    model_class = get_model_class("simple_audio_cnn")
    print(f"simple_audio_cnn registered: {model_class is not None}")
    print(f"   Class: {model_class.__name__ if model_class else 'N/A'}")

    # Note: build_model() requires a ModelSpec to be registered, not just a model class.
    # For custom models without ModelSpecs, instantiate directly.
    # Custom models can be used standalone or with probes attached via
    # build_probe_from_config() with base_model or input_dim
    model = SimpleAudioCNN(device=device, num_classes=5)
    print(f"\nDirect instantiation example: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_input = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input: {dummy_input.shape} -> Output: {output.shape}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Key Takeaways")
    print("=" * 50)
    print("""
When to Register:
- ✅ Required: Using build_model() or build_model_from_spec() with ModelSpecs
- ✅ Required: Loading models from YAML configuration files
- ❌ NOT needed: Direct instantiation (MyModel(device="cpu", num_classes=10))
- ❌ NOT needed: Standalone usage or attaching probes directly

Registration Process:
- Use @register_model_class decorator to register custom models
- Models must inherit from ModelBase
- Define 'name' class attribute for registration
- build_model() requires both a registered ModelSpec AND a registered model class

Alternative (No Registration):
- For custom models without ModelSpecs, instantiate directly
- Attach probes using build_probe_from_config() with base_model for online mode or input_dim for offline mode
- See docs/custom_model_registration.md for more details
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Model Registration Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
