import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class LinearProbe(nn.Module):
    """
    A linear probing model that takes embeddings from any model and produces classification outputs.
    
    Args:
        input_dim: Dimension of the input embeddings
        num_classes: Number of output classes
        device: Device to run the model on
    """
    def __init__(self, input_dim: int, num_classes: int, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.classifier = nn.Linear(input_dim, num_classes).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear probe.
        
        Args:
            x: Input embeddings of shape (batch_size, input_dim)
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        return self.classifier(x) 