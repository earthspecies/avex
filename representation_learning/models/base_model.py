import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, Any

from representation_learning.data.audio_utils import AudioProcessor

class ModelBase(nn.Module):
    def __init__(self, device: str, audio_config: Optional[Dict[str, Any]] = None):
        super(ModelBase, self).__init__()
        self.device = device
        
        # Initialize audio processor if config is provided
        self.audio_processor = AudioProcessor(audio_config) if audio_config else None

    def prepare_inference(self):
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def prepare_train(self):
        self.model.train()
        self.model = self.model.to(self.device)

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio input using the configured audio processor.
        Subclasses can override this method to implement custom audio processing.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps)
            
        Returns:
            Processed audio tensor
        """
        if self.audio_processor is not None:
            x = self.audio_processor(x)
        return x

    def batch_inference(self, batched_samples):
        embeds = []
        for batch in tqdm(batched_samples, desc=" processing batches", position=0, leave=False):
            # Process audio if needed
            batch = self.process_audio(batch)
            
            embedding = self.__call__(batch)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            embeds.append(embedding)
        return torch.cat(embeds, axis=0)
