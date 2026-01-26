import torch

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


if __name__ == "__main__":
    test_efficientnet()
