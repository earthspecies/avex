import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from representation_learning.data.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


class ModelBase(nn.Module):
    def __init__(
        self, device: str, audio_config: Optional[Dict[str, Any]] = None
    ) -> None:
        super(ModelBase, self).__init__()
        self.device = device

        # Initialize audio processor if config is provided
        self.audio_processor = AudioProcessor(audio_config) if audio_config else None

    def prepare_inference(self) -> None:
        self.model.eval()
        self.model = self.model.to(self.device)

    def prepare_train(self) -> None:
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
        return x.to(self.device)

    def batch_inference(self, batched_samples: torch.Tensor) -> torch.Tensor:
        """
        Perform batch inference on input samples.

        Args:
            batched_samples: Input tensor of shape (batch_size, time_steps)

        Returns:
            Concatenated embeddings tensor
        """
        embeds: List[torch.Tensor] = []
        for batch in tqdm(
            batched_samples, desc=" processing batches", position=0, leave=False
        ):
            # Process audio if needed
            batch = self.process_audio(batch)

            embedding = self.__call__(batch)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            embeds.append(embedding)
        return torch.cat(embeds, axis=0)

    def extract_embeddings(self, x: torch.Tensor, layers: List[str]) -> torch.Tensor:
        """
        Extract embeddings from specified layers of the model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav' and 'padding_mask'
            layers: List of layer names to extract embeddings from

        Returns
        -------
        torch.Tensor
            Concatenated embeddings from the requested layers.

        Raises
        ------
        ValueError
            If none of the supplied *layers* are found in the model.
        """
        embeddings = []

        def hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Capture the tensor without detaching so gradients can propagate
            embeddings.append(output)  # noqa: F821

        hooks = []
        try:
            for name, module in self.named_modules():
                if name in layers:
                    hooks.append(module.register_forward_hook(hook_fn))

            # Forward pass (no torch.no_grad to allow fine-tuning when requested)
            if isinstance(x, dict):
                # If input is a dictionary, extract raw_wav and padding_mask
                raw_wav = x["raw_wav"]
                padding_mask = x["padding_mask"]
                self(raw_wav, padding_mask)
            else:
                # For backward compatibility, create a padding mask of ones
                padding_mask = torch.ones(x.size(0), x.size(1), device=x.device)
                if self.__class__.__name__ == "CLIPModel":
                    dummy_text = ["" for _ in range(x.size(0))]
                    self(x, dummy_text, padding_mask)
                else:
                    self(x, padding_mask)

            # Concatenate embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {layers}")

            # Process embeddings
            result = []
            for emb in embeddings:
                # Flatten while keeping on GPU
                flattened = emb.flatten(start_dim=1)
                result.append(flattened)

            return torch.cat(result, dim=1)

        finally:
            # Ensure hooks are always removed
            for hook in hooks:
                hook.remove()
            # Clear any remaining references
            del embeddings
