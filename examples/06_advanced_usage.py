"""
Example 6: Advanced Usage Patterns

This example demonstrates:
- Complex model architectures
- Advanced configuration patterns
- Integration with external libraries
- Performance optimization techniques
- Error handling and debugging
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from representation_learning import (
    create_model,
    describe_model,
    list_models,
    register_model,
    register_model_class,
)
from representation_learning.configs import AudioConfig, ModelSpec
from representation_learning.models.base_model import ModelBase


# Advanced model with attention mechanism
@register_model_class
class AttentionAudioModel(ModelBase):
    """Advanced audio model with attention mechanism."""

    name = "attention_audio_model"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: dict = None,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(1000, d_model)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        # Feed-forward networks
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.to(device)

    def _create_positional_encoding(
        self, max_len: int, d_model: int
    ) -> torch.nn.Parameter:
        """Create sinusoidal positional encoding.

        Returns:
            torch.nn.Parameter: Positional encoding parameter.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass with attention mechanism.

        Returns:
            torch.Tensor: Model output.
        """
        batch_size, seq_len = x.shape

        # Project input
        x = x.unsqueeze(-1)  # Add feature dimension
        x = self.input_projection(x)

        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            pos_enc = self.pos_encoding.repeat(
                1, (seq_len // self.pos_encoding.size(1)) + 1, 1
            )
            pos_enc = pos_enc[:, :seq_len, :]

        x = x + pos_enc.to(x.device)

        # Apply attention layers
        for attention, layer_norm, ffn in zip(
            self.attention_layers, self.layer_norms, self.ffns, strict=False
        ):
            # Self-attention
            attn_output, _ = attention(x, x, x, key_padding_mask=padding_mask)
            x = layer_norm(x + attn_output)

            # Feed-forward
            ffn_output = ffn(x)
            x = layer_norm(x + ffn_output)

        # Global attention pooling
        # Create a learnable query for global pooling
        global_query = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        pooled_output, _ = self.global_attention(
            global_query, x, x, key_padding_mask=padding_mask
        )
        pooled_output = pooled_output.squeeze(1)

        # Classify
        output = self.classifier(pooled_output)
        return output

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The embedding dimension.
        """
        return self.d_model


# Model with residual connections
@register_model_class
class ResidualAudioModel(ModelBase):
    """Audio model with residual connections."""

    name = "residual_audio_model"

    def __init__(
        self,
        device: str,
        num_classes: int,
        audio_config: dict = None,
        hidden_dims: list = None,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        if hidden_dims is None:
            hidden_dims = [128, 256, 512, 256, 128]
        self.hidden_dims = hidden_dims

        # Build residual blocks
        self.blocks = nn.ModuleList()
        input_dim = 1

        for _i, hidden_dim in enumerate(hidden_dims):
            block = ResidualBlock(input_dim, hidden_dim)
            self.blocks.append(block)
            input_dim = hidden_dim

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )

        self.to(device)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Classify
        output = self.classifier(x)
        return output

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The embedding dimension.
        """
        return self.hidden_dims[-1]


class ResidualBlock(nn.Module):
    """Residual block for audio processing."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


def main() -> None:
    print("🚀 Example 6: Advanced Usage Patterns")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Example 1: Advanced model architectures
    print("\n🏗️ Advanced Model Architectures:")
    try:
        # Test attention model
        print("   Testing Attention Audio Model:")
        model = create_model(
            "attention_audio_model",
            num_classes=15,
            device=device,
            d_model=128,  # Smaller for demo
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
        print(f"   ✅ Created attention model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test with different input lengths
        for length in [1000, 2000, 5000]:
            dummy_input = torch.randn(2, length).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"   Input {dummy_input.shape} -> Output {output.shape}")

        # Test residual model
        print("\n   Testing Residual Audio Model:")
        model = create_model(
            "residual_audio_model",
            num_classes=10,
            device=device,
            hidden_dims=[64, 128, 256, 128, 64],
        )
        print(f"   ✅ Created residual model: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        dummy_input = torch.randn(3, 16000).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input {dummy_input.shape} -> Output {output.shape}")

    except Exception as e:
        print(f"❌ Error with advanced models: {e}")
        import traceback

        traceback.print_exc()

    # Example 2: Complex configuration patterns
    print("\n⚙️ Complex Configuration Patterns:")
    try:
        # Create a complex model specification
        complex_config = ModelSpec(
            name="efficientnet",
            pretrained=False,
            device=device,
            audio_config=AudioConfig(
                sample_rate=22050,
                representation="mel_spectrogram",
                n_mels=256,
                n_fft=2048,
                hop_length=512,
                target_length_seconds=10,
                window_selection="center",
                normalize=True,
            ),
            efficientnet_variant="b3",
        )

        # Register the complex configuration
        register_model("complex_efficientnet", complex_config)
        print("✅ Registered complex configuration")

        # Create model from complex config
        model = create_model("complex_efficientnet", num_classes=20, device=device)
        print(f"✅ Created model from complex config: {type(model).__name__}")

        # Get detailed information
        describe_model("complex_efficientnet", verbose=True)

    except Exception as e:
        print(f"❌ Error with complex configuration: {e}")

    # Example 3: Performance optimization
    print("\n⚡ Performance Optimization:")
    try:
        # Test different model sizes
        model_configs = [
            ("attention_audio_model", {"d_model": 64, "num_layers": 1, "nhead": 2}),
            ("attention_audio_model", {"d_model": 128, "num_layers": 2, "nhead": 4}),
            ("attention_audio_model", {"d_model": 256, "num_layers": 4, "nhead": 8}),
        ]

        test_input = torch.randn(1, 16000).to(device)

        for model_name, config in model_configs:
            try:
                model = create_model(
                    model_name, num_classes=10, device=device, **config
                )
                param_count = sum(p.numel() for p in model.parameters())

                # Time inference
                import time

                model.eval()

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

                avg_time = (end_time - start_time) / 100

                print(f"   {model_name} ({config}):")
                print(f"     Parameters: {param_count:,}")
                print(f"     Inference time: {avg_time * 1000:.2f}ms")

            except Exception as e:
                print(f"   {model_name}: Error - {e}")

    except Exception as e:
        print(f"❌ Error in performance testing: {e}")

    # Example 4: Error handling and debugging
    print("\n🐛 Error Handling and Debugging:")
    try:
        # Test invalid model creation
        print("   Testing error handling:")

        try:
            model = create_model("nonexistent_model", num_classes=10, device=device)
            print("   ❌ Unexpected: Model creation succeeded")
        except Exception as e:
            print(f"   ✅ Expected error: {e}")

        # Test invalid parameters
        try:
            model = create_model(
                "attention_audio_model",
                num_classes=10,
                device=device,
                invalid_param="test",
            )
            print("   ❌ Unexpected: Model creation succeeded with invalid param")
        except Exception as e:
            print(f"   ✅ Expected error with invalid param: {e}")

        # Test model introspection for debugging
        print("\n   Model introspection for debugging:")
        models = list_models()
        for name, _spec in list(models.items())[:2]:  # Test first 2 models
            try:
                print(f"   {name}:")
                describe_model(name, verbose=True)
            except Exception as e:
                print(f"   Error describing {name}: {e}")

    except Exception as e:
        print(f"❌ Error in error handling test: {e}")

    # Example 5: Integration patterns
    print("\n🔗 Integration Patterns:")
    try:
        # Create a model for integration
        model = create_model("residual_audio_model", num_classes=5, device=device)
        model.eval()

        # Simulate batch processing
        batch_size = 8
        audio_length = 16000
        batch_input = torch.randn(batch_size, audio_length).to(device)

        print(f"   Processing batch of {batch_size} samples:")

        # Process in chunks to simulate memory constraints
        chunk_size = 2
        outputs = []

        for i in range(0, batch_size, chunk_size):
            chunk = batch_input[i : i + chunk_size]
            with torch.no_grad():
                chunk_output = model(chunk)
            outputs.append(chunk_output)
            print(
                f"     Processed chunk {i // chunk_size + 1}: "
                f"{chunk.shape} -> {chunk_output.shape}"
            )

        # Combine outputs
        final_output = torch.cat(outputs, dim=0)
        print(f"   Final output shape: {final_output.shape}")

        # Simulate confidence scoring
        probabilities = F.softmax(final_output, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)

        print(f"   Predictions: {predictions.cpu().numpy()}")
        print(f"   Confidence scores: {max_probs.cpu().numpy()}")

    except Exception as e:
        print(f"❌ Error in integration patterns: {e}")

    # Example 6: Model ensemble
    print("\n🎯 Model Ensemble:")
    try:
        # Create multiple models for ensemble
        models = []
        for _i in range(3):
            model = create_model(
                "residual_audio_model",
                num_classes=5,
                device=device,
                hidden_dims=[64, 128, 256, 128, 64],
            )
            model.eval()
            models.append(model)

        print(f"   Created ensemble of {len(models)} models")

        # Test ensemble prediction
        test_input = torch.randn(1, 16000).to(device)
        ensemble_outputs = []

        for i, model in enumerate(models):
            with torch.no_grad():
                output = model(test_input)
                ensemble_outputs.append(output)
            print(f"   Model {i + 1} output: {output.shape}")

        # Average ensemble predictions
        ensemble_output = torch.stack(ensemble_outputs).mean(dim=0)
        print(f"   Ensemble output: {ensemble_output.shape}")

        # Get final prediction
        final_prediction = torch.argmax(ensemble_output, dim=1)
        print(f"   Final prediction: {final_prediction.item()}")

    except Exception as e:
        print(f"❌ Error in model ensemble: {e}")

    print("\n🎉 Advanced usage examples completed!")


if __name__ == "__main__":
    main()
