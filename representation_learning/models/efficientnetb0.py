import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from representation_learning.models.base_model import ModelBase

# EfficientNetB0. Each class should be called "Model."
class Model(ModelBase):
    def __init__(self, num_classes=1000, pretrained=True, device='cuda'):
        # Call initializers for both parent classes.
        ModelBase.__init__(self, device)
        nn.Module.__init__(self)
        
        # Load a pre-trained EfficientNet B0 from torchvision.
        self.model = efficientnet_b0(pretrained=pretrained)
        
        # If you need a different number of output classes than the default 1000,
        # modify the classifier. Here, classifier[1] is the final fully connected layer.
        if num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Forward pass: simply call the underlying EfficientNet.
        return self.model(x)
