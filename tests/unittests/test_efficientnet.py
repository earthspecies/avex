
import torch
from representation_learning.models.base_model import EfficientNet

def test_efficientnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create an instance of your EfficientNet model.
    model = EfficientNet(num_classes=1000, pretrained=True, device=device)
    
    # Prepare the model for inference or training.
    model.prepare_inference()  # or model.prepare_train()
    
    # Create a dummy input (e.g., a batch of images with appropriate size)
    # EfficientNet B0 expects images of at least 224x224 in size.
    dummy_input = torch.randn(8, 3, 224, 224).to(device)
    
    # Perform a forward pass
    outputs = model(dummy_input)
    print("Output shape:", outputs.shape)

if __name__ == '__main__':
    test_efficientnet()
