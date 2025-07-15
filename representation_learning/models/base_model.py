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

        target_device = next(self.parameters()).device
        return x.to(target_device)

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory optimization.

        Subclasses must implement this method to enable checkpointing
        for their specific architecture.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support gradient checkpointing. "
            f"Please implement the enable_gradient_checkpointing method."
        )

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

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
    ) -> torch.Tensor:
        """
        Extract embeddings from specified layers of the model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav' and 'padding_mask'
            layers: List of layer names to extract embeddings from
            padding_mask: Optional padding mask tensor

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
        output_padding_mask = []

        def hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            nonlocal embeddings  # noqa: F823 – defined in enclosing scope
            # Capture the tensor without detaching so gradients can propagate
            if isinstance(output, dict):  # TODO: hacky - model-specific handling
                embeddings.append(output["x"])
            elif isinstance(output, tuple):
                embeddings.append(output[0])
                output_padding_mask.append(output[1])
            else:
                embeddings.append(output)

        hooks = []
        try:
            for name, module in self.named_modules():
                if name in layers:
                    logger.info(f"Registering forward hook for {name}")
                    hooks.append(module.register_forward_hook(hook_fn))

            # Forward pass (no torch.no_grad to allow fine-tuning when requested)
            if isinstance(x, dict):
                # Input provided as dictionary with explicit padding mask
                raw_wav = x["raw_wav"]
                p_mask = x["padding_mask"]
                self(raw_wav, p_mask)
            else:
                # Tensor input – use provided mask if available, otherwise assume
                # fully-valid signal (all ones).
                if padding_mask is None:
                    padding_mask = torch.zeros(
                        x.size(0), x.size(1), device=x.device, dtype=torch.bool
                    )
                self(x, padding_mask)

            # Concatenate embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {layers}")

            if average_over_time:
                result = []
                for emb in embeddings:
                    if emb.dim() == 2:
                        # Already in correct shape, just append
                        result.append(emb)
                    elif emb.dim() == 3:
                        aggregated = torch.mean(emb, dim=1)
                        result.append(aggregated)
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {emb.dim()}. "
                            f"Expected 2 or 3."
                        )
                return torch.cat(result, dim=1)
            else:
                return embeddings

        finally:
            for hook in hooks:
                hook.remove()
            del embeddings
