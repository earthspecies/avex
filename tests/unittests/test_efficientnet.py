import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.efficientnet import Model as EfficientNet


def test_efficientnet() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of your EfficientNet model.
    model = EfficientNet(num_classes=1000, pretrained=True, device=device)

    # Prepare the model for inference.
    model.eval()
    model.to(device)

    # Create a dummy input (e.g., a batch of images with appropriate size)
    # EfficientNet B0 expects images of at least 224x224 in size.
    dummy_input = torch.randn(8, 3, 224, 224).to(device)

    # Test forward pass without padding_mask (padding_mask defaults to None)
    outputs = model(dummy_input)
    assert outputs.shape == (8, 1000), f"Expected shape (8, 1000), got {outputs.shape}"
    print("Output shape (without padding_mask):", outputs.shape)

    # Test forward pass with explicit padding_mask (should also work)
    padding_mask = torch.zeros(8, 224 * 224, dtype=torch.bool).to(device)
    outputs_with_mask = model(dummy_input, padding_mask)
    assert outputs_with_mask.shape == (8, 1000), f"Expected shape (8, 1000), got {outputs_with_mask.shape}"
    print("Output shape (with padding_mask):", outputs_with_mask.shape)


def test_efficientnet_float64_audio_input() -> None:
    """Test that EfficientNet automatically converts float64 audio to float32.

    This test verifies the fix for issue #145 where EfficientNet would fail
    silently or produce errors when audio input was float64 instead of float32.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create audio config for spectrogram processing
    audio_config = AudioConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window="hann",
        n_mels=128,
        representation="mel_spectrogram",
        normalize=False,
        target_length_seconds=5,
        window_selection="random",
    )

    # Create EfficientNet model with audio config
    model = EfficientNet(
        num_classes=10,
        pretrained=False,  # Use False for faster testing
        device=device,
        audio_config=audio_config,
        return_features_only=True,
    )
    model.eval()

    # Create audio input with float64 dtype (the problematic case)
    # 5 seconds of audio at 16kHz = 80000 samples
    audio_input = torch.randn(2, 80000, dtype=torch.float64).to(device)

    # This should work without errors - the model should automatically convert to float32
    with torch.no_grad():
        output = model(audio_input)

    # Verify output shape is correct (batch, channels, height, width)
    assert output.dtype == torch.float32, "Output should be float32"
    assert len(output.shape) == 4, "Output should be 4D (B, C, H, W)"
    print(f"Successfully processed float64 audio input. Output shape: {output.shape}")


if __name__ == "__main__":
    test_efficientnet()
    test_efficientnet_float64_audio_input()
