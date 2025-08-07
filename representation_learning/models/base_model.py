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

        # Initialize hook management
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hook_outputs: Dict[str, torch.Tensor] = {}
        self._linear_layer_names: List[str] = []
        # Note: _discover_linear_layers() will be called lazily when needed

    def _discover_linear_layers(self) -> None:
        """Discover and cache all linear layer names in the model."""
        self._linear_layer_names = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_layer_names.append(name)
        logger.debug(
            f"Discovered {len(self._linear_layer_names)} linear layers: "
            f"{self._linear_layer_names}"
        )

    def _create_hook_fn(self, layer_name: str) -> callable:
        """Create a hook function for a specific layer.

        Returns:
            callable: A hook function that captures layer outputs.
        """

        def hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Capture the tensor without detaching so gradients can propagate
            logger.debug(f"Hook triggered for layer: {layer_name}")
            if isinstance(output, dict):  # TODO: hacky - model-specific handling
                self._hook_outputs[layer_name] = output["x"]
            elif isinstance(output, tuple):
                self._hook_outputs[layer_name] = output[0]
            else:
                self._hook_outputs[layer_name] = output
            logger.debug(
                f"Captured output for {layer_name}: "
                f"{self._hook_outputs[layer_name].shape}"
            )

        return hook_fn

    def _register_hooks_for_layers(self, layer_names: List[str]) -> None:
        """Register hooks for the specified layers if not already registered."""
        for layer_name in layer_names:
            if layer_name not in self._hooks:
                # Find the module
                module = None
                for name, mod in self.named_modules():
                    if name == layer_name:
                        module = mod
                        break

                if module is not None:
                    hook_fn = self._create_hook_fn(layer_name)
                    self._hooks[layer_name] = module.register_forward_hook(hook_fn)
                    logger.debug(
                        f"Registered hook for layer: {layer_name} on module: {module}"
                    )
                else:
                    logger.warning(f"Layer '{layer_name}' not found in model")

    def _clear_hook_outputs(self) -> None:
        """Clear cached hook outputs."""
        self._hook_outputs.clear()

    def _get_all_linear_layers(self) -> List[str]:
        """Get all linear layer names including the classification layer.

        Returns:
            List[str]: List of linear layer names.
        """
        # Discover linear layers if not already done
        if not self._linear_layer_names:
            self._discover_linear_layers()

        return self._linear_layer_names

    def _cleanup_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
        self._hook_outputs.clear()

    def __del__(self) -> None:
        """Cleanup hooks when the model is destroyed."""
        try:
            self._cleanup_hooks()
        except Exception:
            pass  # Ignore errors during cleanup

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
            layers: List of layer names to extract embeddings from. If 'all' is
                   included, all linear layers in the model will be automatically
                   found and used.
            padding_mask: Optional padding mask tensor
            average_over_time: Whether to average embeddings over time dimension

        Returns
        -------
        torch.Tensor
            Concatenated embeddings from the requested layers.

        Raises
        ------
        ValueError
            If none of the supplied *layers* are found in the model.
        """
        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Handle 'all' case - use cached linear layers
        target_layers = layers.copy()
        if "all" in layers:
            logger.info("'all' specified in layers, using cached linear layers...")
            linear_layer_names = self._get_all_linear_layers()

            if not linear_layer_names:
                logger.warning("No linear layers found in the model")
            else:
                logger.info(f"Using linear layers: {linear_layer_names}")

            # Replace 'all' with the actual linear layer names
            target_layers = [
                layer for layer in layers if layer != "all"
            ] + linear_layer_names
            logger.info(f"Target layers after 'all' expansion: {target_layers}")

        # Register hooks for requested layers (only if not already registered)
        self._register_hooks_for_layers(target_layers)

        try:
            # Forward pass (no torch.no_grad to allow fine-tuning when requested)
            logger.debug(f"Starting forward pass with target layers: {target_layers}")
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

            logger.debug(
                f"Forward pass completed. Hook outputs: "
                f"{list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []
            logger.debug(
                f"Collecting embeddings from {len(target_layers)} target layers"
            )
            for layer_name in target_layers:
                if layer_name in self._hook_outputs:
                    embeddings.append(self._hook_outputs[layer_name])
                    logger.debug(
                        f"Found embedding for {layer_name}: "
                        f"{self._hook_outputs[layer_name].shape}"
                    )
                else:
                    logger.warning(f"No output captured for layer: {layer_name}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {target_layers}")

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
            # Clear hook outputs for next call
            self._clear_hook_outputs()
