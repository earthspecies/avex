"""
Example 5: Embedding Extraction Mode

This example demonstrates loading models without a classification head
for embedding extraction, which is useful for representation learning tasks.

Key use cases:
- Loading pre-trained models for feature extraction
- Building models without classifiers for embedding analysis
- Using models with return_features_only=True

Audio Requirements:
- Each model expects a specific sample rate (defined in model_spec.audio_config.sample_rate)
- Check with: describe_model("model_name") or get_model_spec("model_name").audio_config.sample_rate
- For full reproducibility, resample using librosa with these exact parameters:

    import librosa
    audio_resampled = librosa.resample(
        audio, orig_sr=original_sr, target_sr=target_sr,
        res_type="kaiser_best", scale=True
    )

Output Formats:
- Transformer models (BEATs, EAT) return 3D tensors: (batch, frames/patches, features)
- CNN models (EfficientNet, ResNet) return 4D tensors: (batch, channels, height, width)
"""

import argparse

import torch

from representation_learning import load_model
from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNetModel


def main(device: str = "cpu") -> None:
    """Demonstrate embedding extraction mode."""
    print("Example 5: Embedding Extraction Mode")
    print("=" * 50)

    # =========================================================================
    # Part 1: BEATs NatureLM (Transformer - 3D output)
    # =========================================================================
    print("\nPart 1: BEATs NatureLM (Transformer)")
    print("-" * 50)

    # Load without num_classes - automatically uses return_features_only=True
    model = load_model("beats_naturelm", device=device)
    model.eval()

    print(f"Loaded model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Return features only: {getattr(model, '_return_features_only', 'N/A')}")

    # BEATs expects 16kHz audio
    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)

    print(f"\n   Output shape: {output.shape}")
    print(f"   - Batch size: {output.shape[0]}")
    print(f"   - Number of frames: {output.shape[1]}")
    print(f"   - Feature dimension: {output.shape[2]}")

    # =========================================================================
    # Part 2: EAT (Transformer - 3D output)
    # =========================================================================
    print("\nPart 2: EAT (Transformer)")
    print("-" * 50)

    # This model has a classifier; request embedding output for this example.
    model = load_model("sl_eat_animalspeak_ssl_all", device=device, return_features_only=True)
    model.eval()

    print(f"Loaded model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Return features only: {getattr(model, 'return_features_only', 'N/A')}")

    # EAT expects 16kHz audio
    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)

    print(f"\n   Output shape: {output.shape}")
    print(f"   - Batch size: {output.shape[0]}")
    print(f"   - Number of patches: {output.shape[1]}")
    print(f"   - Feature dimension: {output.shape[2]}")

    # =========================================================================
    # Part 3: EfficientNet (CNN - 4D output)
    # =========================================================================
    print("\nPart 3: EfficientNet (CNN)")
    print("-" * 50)

    audio_config = AudioConfig(
        sample_rate=16000,
        target_length_seconds=5,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
    )
    model = EfficientNetModel(
        num_classes=10,  # Required but ignored when return_features_only=True
        pretrained=False,
        device=device,
        audio_config=audio_config,
        return_features_only=True,
        efficientnet_variant="b0",
    )
    model.eval()

    print(f"Loaded model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Return features only: {getattr(model, 'return_features_only', 'N/A')}")

    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        output = model(dummy_input, padding_mask=None)

    print(f"\n   Output shape: {output.shape}")
    print(f"   - Batch size: {output.shape[0]}")
    print(f"   - Channels: {output.shape[1]}")
    print(f"   - Height: {output.shape[2]}")
    print(f"   - Width: {output.shape[3]}")

    # =========================================================================
    # Part 4: Embedding vs Classification mode comparison
    # =========================================================================
    print("\nPart 4: Embedding vs Classification Mode")
    print("-" * 50)

    # Embedding mode
    embedding_model = load_model("sl_beats_all", return_features_only=True, device=device)
    embedding_model.eval()

    # Classification mode
    classification_model = load_model("sl_beats_all", device=device)
    classification_model.eval()

    # Check for label mapping
    if hasattr(classification_model, "label_mapping") and classification_model.label_mapping:
        index_to_label = classification_model.label_mapping["index_to_label"]
        label_to_index = classification_model.label_mapping["label_to_index"]
        print(f"Label mapping loaded: {len(label_to_index)} classes")

    # Compare outputs
    dummy_input = torch.randn(1, 16000 * 5, device=device)
    with torch.no_grad():
        embedding_output = embedding_model(dummy_input, padding_mask=None)
        classification_output = classification_model(dummy_input, padding_mask=None)

    print(f"\nEmbedding mode output: {embedding_output.shape}")
    print(
        f"   -> (batch={embedding_output.shape[0]}, "
        f"frames={embedding_output.shape[1]}, "
        f"features={embedding_output.shape[2]})"
    )

    print(f"\nClassification mode output: {classification_output.shape}")
    print(f"   -> (batch={classification_output.shape[0]}, num_classes={classification_output.shape[1]})")

    # Show example predictions
    if hasattr(classification_model, "label_mapping") and classification_model.label_mapping:
        probs = torch.softmax(classification_output, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(3, classification_output.shape[-1]), dim=-1)
        print("\nTop-3 predicted classes:")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0], strict=False)):
            label = index_to_label.get(idx.item(), f"Unknown (index {idx.item()})")
            print(f"  {i + 1}. {label}: {prob.item():.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Key Takeaways")
    print("=" * 50)
    print("""
- Load without num_classes for automatic embedding mode
- Or use return_features_only=True explicitly

Output dimensions:
- Transformers (BEATs, EAT): 3D (batch, frames/patches, features)
- CNNs (EfficientNet, ResNet): 4D (batch, channels, height, width)

Use cases:
- Representation learning
- Feature analysis
- Transfer learning
- Downstream task fine-tuning
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Extraction Example")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for model and data (e.g. cpu, cuda, cuda:0)",
    )
    args = parser.parse_args()
    main(device=args.device)
